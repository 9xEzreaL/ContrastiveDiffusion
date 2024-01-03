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
                "mask_type": "all" // "all", "mess", "eff"
            }
        },
Run 
```
   python run.py -c CONFIG -p test
```

1. First model: vanilla palette 

```
    CONFIG: config/local/pain.json
    
    models-> models.model
```

2. Second model: dualE
```
    CONFIG: config/local/pain-prev.json
    
    models-> models.local_prev_guided_network
```

3. Third model: dualE + SPADE (SPADE in encoder and decoder)
```
    CONFIG: config/local/pain-prev-seg-spade.json
    
    models-> models.local_prev_seg_guided_network
```

4. Forth model: dualE + SPADE + contrastive feeatures (Not performance well)
```
    CONFIG: config/local/prev-seg-spade-cls-free.json
    
    models-> models.local_prev_seg_guided_network_free
```

5. Fifth model: concat mask 
```
    CONFIG: config/local/pain-oncat.json
    
    models-> models.model
```

6. 3D model: instantiate from Med-DDPM
```
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
 