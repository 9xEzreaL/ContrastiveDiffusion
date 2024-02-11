# Palette: Image-to-Image Diffusion Models

## Step for started
<br>
Go json file

    "test": {
        "which_dataset": {
            "name": ["data.dataset", "PainDataset"], // import Dataset() class / function(not recommend) from default file
            "args":{
                "data_root": "/media/ExtHDD01/Dataset/OAI_pain/full/a/a/*", // ap image
                "eff_root": "/media/ExtHDD01/Dataset/OAI_pain/full/apeff", // eff mask
                "mean_root": "/media/ExtHDD01/Dataset/OAI_pain/full/apmean_102323", //mean mask
                "mode": "test",
                "mask_type": "all", // "all", "mess", "eff"
                "threshold": [0.03, 0.15] // should be [min, max] in training, 0.06 in testing
            }
        },
Run for training
```
   python run.py -c CONFIG -p train
```

Run for testing
```
   python run.py -c CONFIG -p test
```

1. First model: vanilla palette 

```
    TRAIN:
        CONFIG: config/online/pain.json
    TEST:
        CONFIG: config/local/pain.json
    
    models-> models.model
```

2. Second model: dualE
```    
    TRAIN:
        CONFIG: config/online/pain_prev.json
    TEST:
        CONFIG: config/local/pain_prev.json
    
    models-> models.local_prev_guided_network
```

3. Third model: dualE + SPADE (SPADE in encoder and decoder)
```
    TRAIN:
        CONFIG: config/online/pain_prev_spade_EnD.json
    TEST:
        CONFIG: config/local/pain_prev_spade_EnD.json
    
    models-> models.local_prev_seg_guided_network
```

4. Forth model: dualE + SPADE (SPADE in encoder)
```
    TRAIN:
        CONFIG: config/online/pain_prev_spade.json
    TEST:
        CONFIG: config/local/pain_prev_spade.json
            
    models-> models.local_prev_seg_guided_network_free
```

5. Fifth model: concat mask 
```
    TRAIN:
        CONFIG: config/online/pain_concat.json
    TEST:
        CONFIG: config/local/pain_concat.json
    
    models-> models.model
```

6. 3D model: instantiate from Med-DDPM
```
    TRAIN:
        CONFIG: config/online/3D_model.json
    TEST:
        CONFIG: config/local/3D_model.json
    
    models-> models.3D_network
```




## UNet model Architecture
```
    vanilla palette -> models/guided_diffusion_modules/unet.py 
    dualE -> models/guided_diffusion_modules/guided_unet.py
    dualE + SPADE (SPADE in Encoder + Decoder)-> models/guided_diffusion_modules/guided_spade_EnD_unet.py
    dualE + SPADE + classfier_free -> models/guided_diffusion_modules/guided_spade_EnD_unet_free.py
    concat -> models/guided_diffusion_modules/unet.py
    3D model -> models/guided_diffusion_modules/medddpm_unet.py
```
 