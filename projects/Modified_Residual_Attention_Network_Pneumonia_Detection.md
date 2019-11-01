---
layout: project
title: "Modified Residual Attention Network Pneumonia Detection"
author: Sajaratul Yakin Rubaiat
comments: true
---

___

# The model I tried,


```

Name              Parameter          DataSet                    Accuracy     Dataset size  Local     Modification
                                                                                            ACC
DenseNet121       79,78,856     NHI_Xray_dataset                 81.7 %        112121      19/20   Visualize the activation Layer

ResNet50         2,55,57,032    RSNA_pnumonia_dataset            87.33 %      26684/3000   20/20   Implement Basic reset50 with focal loss and Adam Op.

ResNet50         2,55,57,032    NHI_Xray_dataset(Pnumonia sub)   85.10 %      26684/3000   20/20   Implement Basic reset50 with focal loss and Adam Op.

ResNet101        4,45,49,160    RSNA_pnumonia_dataset            89.16 %         ""        20/20   Implement ResNet101 with focal, Img: 254

ResNet101        4,45,49,160    NHI_Xray_dataset(Pnumonia sub)   87.66 %         ""        20/20   Implement ResNet101 with focal, Img: 254

ResNet(50+101)   7,01,60,192    RSNA_pnumonia_dataset            90.03 %         ""        20/20   Same loss as previous, Train iteration: 50,Img: 512

ResNet(50+101)   7,01,60,192    NHI_Xray_dataset(Pnumonia sub)   86.5 %         ""         20/20   Same loss as previous, Train iteration: 50,Img: 512

EfficientNet     5,50,12,455    RSNA_pnumonia_dataset            89.9 %          ""        18/20   EfficientNet with BCE loss/focal loss

VGG16           12,36,42,856    NHI_Xray_dataset(Pnumonia sub)   82.2 %       5228/624      ""     Implemeant pretrained VGG16 with unfreeze all layer

ResAttention52   31.9 * 10^6    NHI_Xray_dataset(Pnumonia sub)   79.6 %          ""         ""     Basic implemention of ResAttention52

ResAttention36    99,25,490     NHI_Xray_dataset(Pnumonia sub)   85.19 %         ""        20/20   Downgrading in ResAtt52 in resudial block and Attention block, Optimizar: RMSprop, Loss: Softmax

ResAttention36    99,25,490     NHI_Xray_dataset(Pnumonia sub)   88.13 %         ""        20/20   Downgrading in ResAtt52 in resudial block and Attention block, Optimizar: Adam, Loss: Focal

```
 


# Layer 

### Residual Block

Residual block mainly a way to prevent the vanishing gradient. It adds the weight of its first layer with the last layer of the residual block.  There are many ways to structure the residual block,
the most famous is Batchnorm+activationlayer+Conv layer followed by three times.
 
The block that I used: 

```
    x = BatchNormalization()(input)
    x = Activation('relu')(x)
    x = Conv2D(input_channels, kernel_size, padding='same', strides=stride)(x)
    
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(output_channels, (1, 1), padding='same')(x)
    
     x = Add()([x, input])

```

Original Residual Block as Autor proposed : https://arxiv.org/abs/1704.06904

```
    x = BatchNormalization()(input)
    x = Activation('relu')(x)
    x = Conv2D(input_channels, (1, 1))(x)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(input_channels, kernel_size, padding='same', strides=stride)(x)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(output_channels, (1, 1), padding='same')(x)

```

### Attention block

This block is mainly a bottleneck for a network. 

Example is:

```
56 * 56 * 256

28 * 28 * 256

14 * 14 * 256

14 * 14 * 256

28 * 28 * 256 

56 * 56 * 256
```
This is a simplified version of what Attention Block is look like in very simple form

```
    g1 = Conv2D(i_filters,kernel_size = 1)(shortcut) 
    g1 = BatchNormalization()(g1)
    x1 = Conv2D(i_filters,kernel_size = 1)(x) 
    x1 = BatchNormalization()(x1)

    g1_x1 = Add()([g1,x1])
    psi = Activation('relu')(g1_x1)
    psi = Conv2D(1,kernel_size = 1)(psi) 
    psi = BatchNormalization()(psi)
    psi = Activation('sigmoid'))(psi)
 ```
 
My Main Network ResidualAttention36
 
 ```
    input_ = Input(shape=shape)
    x = Conv2D(n_channels, (3, 3), strides=(1, 1), padding='same')(input_) # 224x224
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPool2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x)  # 56x56

    x = residual_block(x, output_channels=n_channels * 4,stride=2)  # 56x56
    x = attention_block(x, encoder_depth=2)  # bottleneck 7x7

    x = residual_block(x, output_channels=n_channels * 8, stride=2)  # 28x28
    x = attention_block(x, encoder_depth=1)  # bottleneck 7x7

    x = residual_block(x, output_channels=n_channels * 16, stride=2)  # 7x7

    pool_size = (x.get_shape()[1].value, x.get_shape()[2].value)
    x = AveragePooling2D(pool_size=pool_size, strides=(1, 1))(x)
    x = Flatten()(x)
    x = Dropout(dropout)(x)
    x = Dense(256, kernel_regularizer=regularizer, activation='relu')(x)
    x = Dropout(dropout)(x)
    x = Dense(16, kernel_regularizer=regularizer, activation='relu')(x)
    x = Dropout(dropout)(x)
    output = Dense(n_classes, kernel_regularizer=regularizer, activation='softmax')(x)

 ```
 
 Original ResidualAttention52 that work well in CIFAR 100 ansd 10
 
 ```
     input_ = Input(shape=shape)
    x = Conv2D(n_channels, (7, 7), strides=(2, 2), padding='same')(input_) # 112x112
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)  # 56x56

    x = residual_block(x, output_channels=n_channels * 4)  # 56x56
    x = attention_block(x, encoder_depth=3)  # bottleneck 7x7

    x = residual_block(x, output_channels=n_channels * 8, stride=2)  # 28x28
    x = attention_block(x, encoder_depth=2)  # bottleneck 7x7

    x = residual_block(x, output_channels=n_channels * 16, stride=2)  # 14x14
    x = attention_block(x, encoder_depth=1)  # bottleneck 7x7

    x = residual_block(x, output_channels=n_channels * 32, stride=2)  # 7x7
    x = residual_block(x, output_channels=n_channels * 32)
    x = residual_block(x, output_channels=n_channels * 32)

    pool_size = (x.get_shape()[1].value, x.get_shape()[2].value)
    x = AveragePooling2D(pool_size=pool_size, strides=(1, 1))(x)
    x = Flatten()(x)
    if dropout:
        x = Dropout(dropout)(x)
    output = Dense(n_classes, kernel_regularizer=regularizer, activation='softmax')(x)

    model = Model(input_, output)
    
 ```

My params: 9,925,490
Original Network Params : 31.9 * 10^6

I changed the loss function to Adam to RMSprop becasue it worked well in my case.

# Accuracy: 

## Data Source: https://data.mendeley.com/datasets/rscbjbr9sj/2

```
          Name             Model            Accuracy
        CheXpert          DenseNet121       0.7680
        Yao et al.        ResNet152         0.713
        Wang et al. (2017) -------          0.633
        Kaggle kernel      VGG16            0.826
        Our network    ResidualAttention32  0.851
        
```


        
