/**
 * AVP Latent — Pi extension
 *
 * Bridges Pi to a persistent AVP FastMCP daemon (see `avp.server.daemon`).
 * Pi has no MCP client, so this extension talks to the daemon's plain-HTTP
 * mirror at `/api/*` and exposes two model-callable tools:
 *
 *   - `latent_think`    → runs latent thinking, returns a `context_id`
 *   - `latent_generate` → generates text, optionally from a `context_id`
 *
 * Latent state (KV-cache / hidden state) never travels through the model
 * context. It stays resident in the daemon and is referenced by the short id,
 * so sub-agents can hand each other `context_id`s instead of long text.
 *
 * Configuration (environment variables):
 *   AVP_DAEMON_URL       Base URL of the daemon   (default http://127.0.0.1:8765)
 *   AVP_DAEMON_TOKEN     Bearer token, if the daemon was started with one
 *   AVP_DAEMON_AUTOSTART Spawn the daemon on session start ("1" to enable)
 *   AVP_DAEMON_STOP_ON_EXIT Kill an autostarted daemon on session shutdown ("1")
 *   AVP_DAEMON_CMD       Override the autostart command
 *
 * @module
 */

import { spawn, type ChildProcess } from "node:child_process";
import { Type } from "@earendil-works/pi-ai";
import {
	defineTool,
	type ExtensionAPI,
	type ExtensionContext,
} from "@earendil-works/pi-coding-agent";

type Json = Record<string, unknown>;

const config = {
	baseUrl: (process.env.AVP_DAEMON_URL ?? "http://127.0.0.1:8765").replace(/\/+$/, ""),
	token: process.env.AVP_DAEMON_TOKEN ?? "",
	autostart: process.env.AVP_DAEMON_AUTOSTART === "1",
	stopOnExit: process.env.AVP_DAEMON_STOP_ON_EXIT === "1",
	command: process.env.AVP_DAEMON_CMD ?? "",
};

let daemonChild: ChildProcess | undefined;
let starting: Promise<void> | undefined;

function sleep(ms: number): Promise<void> {
	return new Promise((resolve) => setTimeout(resolve, ms));
}

function authHeaders(): Record<string, string> {
	const headers: Record<string, string> = { "content-type": "application/json" };
	if (config.token) headers.authorization = `Bearer ${config.token}`;
	return headers;
}

/** POST/GET JSON to the daemon, raising on non-2xx. */
async function callDaemon(
	path: string,
	init: RequestInit,
	signal?: AbortSignal,
): Promise<Json> {
	let response: Response;
	try {
		response = await fetch(`${config.baseUrl}${path}`, {
			...init,
			headers: { ...authHeaders(), ...(init.headers ?? {}) },
			signal,
		});
	} catch (error) {
		throw new Error(
			`AVP daemon unreachable at ${config.baseUrl}. Start it with ` +
				`\`avp-server\` or set AVP_DAEMON_AUTOSTART=1. (${String(error)})`,
		);
	}

	const raw = await response.text();
	let body: Json = {};
	if (raw) {
		try {
			body = JSON.parse(raw) as Json;
		} catch {
			body = { error: raw };
		}
	}
	if (!response.ok) {
		const message = typeof body.error === "string" ? body.error : `HTTP ${response.status}`;
		const code = typeof body.code === "string" ? ` (${body.code})` : "";
		throw new Error(`AVP daemon error: ${message}${code}`);
	}
	return body;
}

async function isHealthy(): Promise<boolean> {
	try {
		const response = await fetch(`${config.baseUrl}/health`, {
			signal: AbortSignal.timeout(1500),
		});
		return response.ok;
	} catch {
		return false;
	}
}

/** Start the daemon detached, then wait for its health endpoint. */
async function startDaemon(ctx: ExtensionContext): Promise<void> {
	if (starting) return starting;
	starting = (async () => {
		const commandLine = config.command || "avp-server";
		const [command, ...args] = commandLine.split(/\s+/);
		ctx.ui.notify(`AVP: starting daemon → ${commandLine}`, "info");
		daemonChild = spawn(command, args, {
			detached: true,
			stdio: "ignore",
			env: process.env,
		});
		daemonChild.unref();

		for (let attempt = 0; attempt < 120; attempt += 1) {
			if (await isHealthy()) return;
			await sleep(500);
		}
		throw new Error(
			"AVP daemon did not become healthy. Check the daemon logs or start it manually.",
		);
	})().finally(() => {
		starting = undefined;
	});
	return starting;
}

/** Ensure the daemon is reachable, optionally autostarting it. */
async function ensureReady(ctx: ExtensionContext): Promise<void> {
	if (await isHealthy()) return;
	if (!config.autostart) {
		throw new Error(
			`AVP daemon not reachable at ${config.baseUrl}. ` +
				"Run `avp-server` (pip install \"avp[mcp]\") or set AVP_DAEMON_AUTOSTART=1.",
		);
	}
	await startDaemon(ctx);
}

const latentThinkTool = defineTool({
	name: "latent_think",
	label: "Latent Think",
	description:
		"Think about a prompt on a self-hosted model and return a short context_id. " +
		"The model's latent state stays resident in the AVP daemon under that id. " +
		"Pass the id to latent_generate so another agent/model continues from the " +
		"same internal state instead of re-reading text. Prefer output=\"hidden_state\" " +
		"when the follow-up model differs (saves VRAM).",
	promptSnippet: "latent_think(prompt, model?, steps?) → context_id",
	executionMode: "sequential",
	parameters: Type.Object({
		prompt: Type.String({ description: "Prompt to think about." }),
		model: Type.Optional(
			Type.String({ description: "Model id; defaults to the daemon's default model." }),
		),
		steps: Type.Optional(
			Type.Number({ description: "Latent thinking steps (default 20).", minimum: 0 }),
		),
		output: Type.Optional(
			Type.Union([Type.Literal("auto"), Type.Literal("kv_cache"), Type.Literal("hidden_state")], {
				description: "Payload to keep resident. Default 'auto' (full KV-cache).",
			}),
		),
		ttl: Type.Optional(
			Type.Number({ description: "Seconds before the context expires (daemon default 300)." }),
		),
		context_id: Type.Optional(
			Type.String({ description: "Continue a prior same-model context instead of starting fresh." }),
		),
	}),
	async execute(_toolCallId, params, signal, _onUpdate, ctx) {
		await ensureReady(ctx);
		const body = await callDaemon(
			"/api/latent_think",
			{ method: "POST", body: JSON.stringify(params) },
			signal,
		);
		const contextId = String(body.context_id ?? "");
		const text =
			`context_id=${contextId}\n` +
			`model=${String(body.source_model_id ?? "?")} ` +
			`steps=${String(body.num_steps ?? "?")} ` +
			`payload=${String(body.payload_type ?? "?")} ` +
			`seq_len=${String(body.seq_len ?? "?")}`;
		return { content: [{ type: "text", text }], details: body };
	},
});

const latentGenerateTool = defineTool({
	name: "latent_generate",
	label: "Latent Generate",
	description:
		"Generate text, optionally conditioned on a context_id from latent_think. " +
		"Same-model contexts reuse the full KV-cache; contexts from another model are " +
		"projected cross-model inside the daemon. Omit context_id and pass steps>0 to " +
		"think-and-generate on the target model, or leave both unset for plain generation.",
	promptSnippet: "latent_generate(prompt, model?, context_id?) → text",
	executionMode: "sequential",
	parameters: Type.Object({
		prompt: Type.String({ description: "Prompt for generation (the answer request)." }),
		model: Type.Optional(
			Type.String({ description: "Target model id; defaults to the daemon's default model." }),
		),
		context_id: Type.Optional(
			Type.String({ description: "Handle returned by latent_think." }),
		),
		steps: Type.Optional(
			Type.Number({ description: "Think steps when no context_id is given (default 0).", minimum: 0 }),
		),
		max_new_tokens: Type.Optional(Type.Number({ description: "Generation cap.", minimum: 1 })),
		temperature: Type.Optional(Type.Number({ description: "Sampling temperature (default 0.7)." })),
		top_p: Type.Optional(Type.Number({ description: "Nucleus sampling threshold (default 0.95)." })),
		do_sample: Type.Optional(Type.Boolean({ description: "Sample (true) or greedy decode (false)." })),
		store_context: Type.Optional(
			Type.Boolean({ description: "When steps>0, also store the produced context and return its id." }),
		),
	}),
	async execute(_toolCallId, params, signal, _onUpdate, ctx) {
		await ensureReady(ctx);
		const body = await callDaemon(
			"/api/latent_generate",
			{ method: "POST", body: JSON.stringify(params) },
			signal,
		);
		const text = typeof body.text === "string" ? body.text : JSON.stringify(body);
		return { content: [{ type: "text", text }], details: body };
	},
});

export default function (pi: ExtensionAPI): void {
	pi.on("session_start", async (_event, ctx) => {
		if (!config.autostart) return;
		try {
			await ensureReady(ctx);
		} catch (error) {
			if (ctx.hasUI) ctx.ui.notify(`AVP: ${String(error)}`, "warning");
		}
	});

	pi.on("session_shutdown", async () => {
		if (daemonChild && config.stopOnExit) {
			try {
				daemonChild.kill();
			} catch {
				// best effort
			}
			daemonChild = undefined;
		}
	});

	pi.registerTool(latentThinkTool);
	pi.registerTool(latentGenerateTool);

	pi.registerCommand("avp-status", {
		description: "Show AVP daemon status: loaded models and resident contexts",
		handler: async (_args, ctx) => {
			try {
				const status = await callDaemon("/api/status", { method: "GET" });
				const models = Array.isArray(status.models) ? status.models.length : 0;
				const registry = (status.registry ?? {}) as Json;
				const active = String(registry.active_count ?? "?");
				const message = `AVP daemon at ${config.baseUrl}: ${models} model(s), ${active} context(s)`;
				if (ctx.hasUI) ctx.ui.notify(message, "info");
			} catch (error) {
				if (ctx.hasUI) ctx.ui.notify(`AVP: ${String(error)}`, "error");
			}
		},
	});
}
