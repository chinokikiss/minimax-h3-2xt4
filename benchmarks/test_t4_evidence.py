"""Evidence acceptance must not turn missing/partial/device-mismatched runs into passes."""
import json
from pathlib import Path
import tempfile
import unittest
from benchmarks.summarize_t4_native_validation import classify


class EvidenceAcceptance(unittest.TestCase):
    def write(self,root,name,value):
        target=root/name;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text(json.dumps(value),encoding='utf-8')

    def test_cpu_linux_pass_is_not_gpu_qualification(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);gpu=root/'gpu';gpu.mkdir();cpu=root/'cpu'
            self.write(cpu,'native_profile/linux_pread.json',{'status':'passed'})
            result=classify(gpu,cpu)
            self.assertFalse(result['physical_t4_evidence'])
            statuses=result['validation_dimensions']
            self.assertEqual(statuses['T09']['status'],'passed_cpu_linux_scope')
            self.assertEqual(statuses['T01']['status'],'partial_or_missing')

    def test_flow_requires_distinct_t4s_both_ranks_and_both_reference_modes(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);cpu=root/'cpu';cpu.mkdir()
            devices=[{'name':'Tesla T4','capability':[7,5],'uuid':f'GPU-{i}'} for i in range(2)]
            self.write(root,'batch_results.json',{'hardware':{'devices':devices}})
            for name in ('te_flow','dit_full_flow','dit_late_lora_flow','dit_mid_flow'):
                for rank in (0,1):
                    self.write(root,f'{name}_rank{rank}.json',{'status':'passed','rank':rank,'cases':[]})
            self.assertEqual(classify(root,cpu)['validation_dimensions']['T01']['status'],'partial_or_missing')
            for name in ('te_flow','dit_full_flow','dit_late_lora_flow','dit_mid_flow'):
                for rank in (0,1):
                    self.write(root,f'{name}_rank{rank}.json',{'status':'passed','rank':rank,'cases':[
                        {'name':n,'dtype':'test','divisor':1} for n in ('native_fp32_flow','fp16_fold_c64')]})
            self.assertEqual(classify(root,cpu)['validation_dimensions']['T01']['status'],'executed_component_scope_accuracy_not_accepted')
            devices[1]['uuid']=devices[0]['uuid']
            self.write(root,'batch_results.json',{'hardware':{'devices':devices}})
            self.assertFalse(classify(root,cpu)['physical_t4_evidence'])
            devices[1]['uuid']='GPU-1';devices[1]['capability']=[12,0]
            self.write(root,'batch_results.json',{'hardware':{'devices':devices}})
            self.assertFalse(classify(root,cpu)['physical_t4_evidence'])


if __name__=='__main__':unittest.main()
