
PYTHON = /home/tamaturgo/desafio-fpf/.venv/bin/python

.PHONY: test test-unit test-integration test-api test-coverage test-watch clean-test install-test-deps generate-test-images

generate-test-images:
	cd tests && $(PYTHON) generate_test_images.py

test:
	$(PYTHON) -m pytest tests/ -v

test-unit:
	$(PYTHON) -m pytest tests/unit/ -v

test-integration:
	$(PYTHON) -m pytest tests/integration/ -v -m integration

test-api:
	$(PYTHON) -m pytest tests/test_api_routes.py -v

test-coverage:
	$(PYTHON) -m pytest tests/ --cov=src --cov-report=term-missing --cov-report=html

test-vision:
	python -m pytest tests/test_vision_processor.py tests/test_yolo_detector.py tests/test_qr_decoder.py -v

test-storage:
	python -m pytest tests/test_result_storage.py -v

test-celery:
	python -m pytest tests/test_celery_tasks.py -v

test-watch:
	python -m pytest tests/ -v --tb=short -f

test-fast:
	python -m pytest tests/ -m "not slow" -v

clean-test:
	rm -rf tests/htmlcov/
	rm -rf tests/__pycache__/
	rm -rf .pytest_cache/
	rm -rf .coverage
	find tests/ -name "*.pyc" -delete
	find tests/ -name "__pycache__" -type d -exec rm -rf {} +

install-test-deps:
	pip install pytest pytest-cov pytest-mock pytest-asyncio httpx

lint-tests:
	flake8 tests/ --max-line-length=100 --ignore=E203,W503

format-tests:
	black tests/

check-imports-tests:
	python -c "import ast, sys; [print(f'Unused import in {f}') for f in sys.argv[1:] if any(isinstance(node, ast.Import) for node in ast.walk(ast.parse(open(f).read())))]" tests/*.py

setup-dev: install-test-deps generate-test-images
	@echo "Setup de desenvolvimento concluído"
	@echo "Execute 'make test' para rodar os testes"

quality-report: test-coverage lint-tests
	@echo "Relatório de qualidade gerado"
	@echo "Cobertura: tests/htmlcov/index.html"

help:
	@echo "Comandos disponíveis:"
	@echo "  generate-test-images - Gera imagens de teste sintéticas"
	@echo "  test              - Executa todos os testes"
	@echo "  test-unit         - Executa testes unitários"
	@echo "  test-integration  - Executa testes de integração"
	@echo "  test-api          - Executa testes de API"
	@echo "  test-coverage     - Executa testes com relatório de cobertura"
	@echo "  test-vision       - Executa testes de visão computacional"
	@echo "  test-storage      - Executa testes de storage"
	@echo "  test-celery       - Executa testes de tarefas Celery"
	@echo "  test-watch        - Executa testes em modo watch"
	@echo "  test-fast         - Executa testes rápidos"
	@echo "  clean-test        - Limpa arquivos de teste"
	@echo "  install-test-deps - Instala dependências de teste"
	@echo "  setup-dev         - Setup completo para desenvolvimento"
	@echo "  quality-report    - Gera relatório de qualidade"