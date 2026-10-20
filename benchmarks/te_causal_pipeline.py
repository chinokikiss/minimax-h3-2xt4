"""Unchanged physical P0 requests with selected warm TE and OS block evidence."""
import os
os.environ['MINIMAX_H3_RESIDENCY_POLICY']='p0'
import residency_pipeline_ablation as trial
import te_block_telemetry as telemetry


original=trial.pipeline.tracing.Recorder


class BlockRecorder(original):
    def stop(self):
        try:return super().stop()
        finally:telemetry.close(self)


trial.pipeline.tracing.Recorder=BlockRecorder
trial.pipeline.tracing.install=telemetry.install
if __name__=='__main__':trial.pipeline.main()
