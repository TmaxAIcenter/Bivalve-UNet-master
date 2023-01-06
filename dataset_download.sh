wget https://github.com/TmaxAIcenter/Bivalve-UNet-master/releases/download/chackpoint/FU_Unet_Checkpoint_epoch200.pth
mv FU_Unet_Checkpoint_epoch200.pth checkpoints/FU_checkpoints/unet/
wget https://github.com/TmaxAIcenter/Bivalve-UNet-master/releases/download/chackpoint/MA_Unet_Checkpoint_epoch200.pth
mv MA_Unet_Checkpoint_epoch200.pth checkpoints/MA_checkpoints/unet/
mkdir Dataset
cd Dataset
mkdir FU
cd FU
wget https://github.com/TmaxAIcenter/Bivalve-UNet-master/releases/download/dataset/FU_test.zip
unzip FU_test.zip
rm FU_test.zip
cd ..
mkdir MA
cd MA
wget https://github.com/TmaxAIcenter/Bivalve-UNet-master/releases/download/dataset/MA_test.zip
unzip MA_test.zip
rm MA_test.zip
