.PHONY: train validate-run sync-context commit-run agent-propose status repro

train:
	python cli.py train --config $(CONFIG) --run-id $(RUN_ID)

validate-run:
	python -m scripts.validate_run --run-id $(RUN_ID)

sync-context:
	python -m scripts.sync_context

commit-run:
	python -m scripts.commit_run --run-id $(RUN_ID)

agent-propose:
	python cli.py agent run --agent training_agent --run-id $(RUN_ID) --payload @$(PAYLOAD)

status:
	python cli.py status

repro:
	python cli.py repro --run-id $(RUN_ID)

# Cron example (daily sweep, same box):
# 0 2 * * *  cd /repo && make train CONFIG=configs/sweep_w_scale.yaml RUN_ID=nightly_$$(date +\%s) >> logs/cron.log 2>&1
