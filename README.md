# Palette: Image-to-Image Diffusion Models

## Step for started
<br>
Go data/XXX_dataset.py find class "PainDataset" correct path 

```
   python run.py -c CONFIG -p train
   ```

1. First model: vanilla palette 

```
    CONFIG: config/local/pain.json
```

3. Second model: dualE
```
    CONFIG: config/local/pain-prev.json
```

5. Third model: dualE + SPADE
```
    CONFIG: config/local/pain-prev-seg-spade.json
```

6. Forth model: dualE + SPADE + contrastive feeatures
```
    CONFIG: config/local/prev-seg-spade-cls-free.json
```
