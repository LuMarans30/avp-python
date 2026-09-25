.PHONY: proto test lint clean install

# Regenerate avp_pb2.py. Keep grpcio-tools on the 5.29 gencode line
# (>=1.71.2,<1.72): the autogen extra pins protobuf to 5.29.x, so a
# newer protoc would emit gencode the runtime refuses to load.
proto:
	python -m grpc_tools.protoc \
		-I proto \
		--python_out=src/avp \
		proto/avp.proto

install:
	pip install -e ".[dev]"

test:
	pytest tests/ -v

lint:
	ruff check src/ tests/

clean:
	python -c "import shutil, glob, os; [shutil.rmtree(p, ignore_errors=True) for p in ['build', 'dist', '.pytest_cache', '__pycache__'] + glob.glob('*.egg-info')]; [os.remove(p) for p in glob.glob('**/*.pyc', recursive=True)]"
