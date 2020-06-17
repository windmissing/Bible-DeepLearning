# EBGAN

EBGAN = Energy-based GAN

使用auto-encoder作为D  
![](/assets/images/GAN/26.png)   

原理：  
一张图如果越成recontruct，则说明它的质量越高。  
优点：可以pre-train D。这样G在一开始就很强，使的G在初期提升很快。  

# GAN-LSGAN

loss-sensitive GAN
![](/assets/images/GAN/27.png)   
