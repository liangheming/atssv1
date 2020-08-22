# ATSS_FCOS_Lite

this repository is an unofficial implement of atss_fcos_lite,we train this model on coco using 4 gpus 8images per gpu.
the resolution of image is 640 x 640.we get this resolution by fix the long side to be 640px and keep the width/height 
ratio.finally we pad the short side to get the square shape.some data augmentation like color jitter,scale jitter,rand 
geometric transformation,mosaic augmentation,rand noise were used on training.the final map on COCO is 36.6

## performance on COCO
```shell script
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.366
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.555
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.388
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.174
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.412
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.527
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.305
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.475
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.504
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.260
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.582
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.685
```
