"""Select extra CPU observers without modifying the validated request runner."""
import causal_pipeline_audit as pipeline
import cpu_causal_trace as cpu

pipeline.tracing.Recorder=cpu.CPURecorder
pipeline.tracing.install=cpu.install

if __name__=='__main__':pipeline.main()
