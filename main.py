from processors.retina.ddp_apex_processor import DDPApexProcessor
from processors.fcos.ddp_apex_processor import DDPApexProcessor

# python -m torch.distributed.launch --nproc_per_node=4 main.py

if __name__ == '__main__':
    # processor = DDPApexProcessor(cfg_path="config/retina.yaml")
    processor = DDPApexProcessor(cfg_path="config/focs.yaml")
    processor.run()
