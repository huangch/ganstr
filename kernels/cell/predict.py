import glob
import pandas as pd
import os
# from PIL import Image
# import argparse
import numpy as np
import csv
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from models.discriminator import Discriminator
from utils import SingleCellImageDataset
from tqdm import tqdm

IMAGE_SIZE = 56

def predict(opt):
    cuda = True if torch.cuda.is_available() else False
    
    model = torch.load(opt.model_file)
    param_json_data = model['parameters']
    
    # Prepare image data
    image_filepath_list = []
    image_uuid_list = []
    existing_image_fileext = None
    
    image_filename_list = glob.glob(os.path.join(opt.image_folder, '*','*'))
    
    for image_filename in image_filename_list:
        image_filepath, image_filename = os.path.split(image_filename)
        image_uuid, image_fileext = os.path.splitext(image_filename)
    
        assert existing_image_fileext == None or existing_image_fileext == image_fileext
        existing_image_fileext = image_fileext
    
        image_filepath_list.append(image_filepath)
        image_uuid_list.append(image_uuid)
    
    image_data = pd.DataFrame({'image_filepath': image_filepath_list, 'image_uuid':image_uuid_list})
    image_data.set_index('image_uuid', inplace=True)
    
    # Curate image and transcript data by cell UUID
    test_uuid_list = [value for value in image_data.index]
    
    # Create image data according to training, validation and test uuids.
    test_image_data = image_data.reindex(test_uuid_list)
    
    # Create datasets
    test_transformed_dataset = SingleCellImageDataset(test_image_data, None, None, image_fileext=existing_image_fileext, image_size=param_json_data['parameters']['image_size'], image_mean=param_json_data['image_mean'], image_std=param_json_data['image_std'], trns_count_per_cell=param_json_data['trns_count_per_cell'])
    
    # Create dataloaders
    testDataLoader = DataLoader(test_transformed_dataset, param_json_data['parameters']['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
      
    # Initialize generator and discriminator
    if param_json_data['parameters']['model'] == 'ae':
        from models.ae import AE
        generator = AE(geneNum=param_json_data['gene_num'], latentDim=param_json_data['parameters']['latent_dim'])
        
    elif param_json_data['parameters']['model'] == 'vae':
        from models.vae import VAE
        generator = VAE(geneNum=param_json_data['gene_num'], latentDim=param_json_data['parameters']['latent_dim'])
        
    elif param_json_data['parameters']['model'] == 'vqvae':
        from models.vqvae import VQVAE
        generator = VQVAE(gene_num=param_json_data['gene_num'], latent_dim=param_json_data['parameters']['latent_dim'], num_embeddings = 1024, commitment_cost = 0.25, decay = 0.99)
        
    elif param_json_data['parameters']['model'] == 'vit':
        from models.vit import ViT
        generator = ViT(image_size = IMAGE_SIZE,
                        patch_size=7,
                        num_classes = param_json_data['gene_num'],
                        dim = 512,
                        depth = 6,
                        heads = 8,
                        mlp_dim = 512,
                        dropout = 0.1,
                        emb_dropout = 0.1) 
     
    elif param_json_data['parameters']['model'] == 'vit_small':
        from models.vit import ViT
        generator = ViT(image_size=IMAGE_SIZE, 
                        patch_size=7, 
                        num_classes = param_json_data['gene_num'],
                        dim = 512,
                        depth = 6,
                        heads = 8,
                        mlp_dim = 512,
                        dropout = 0.1,
                        emb_dropout = 0.1) 
           
    elif param_json_data['parameters']['model'] == 'vit_tiny':
        from models.vit import ViT
        generator = ViT(image_size = IMAGE_SIZE,
                        patch_size = 7,
                        num_classes = param_json_data['gene_num'],
                        dim = 512,
                        depth = 4,
                        heads = 6,
                        mlp_dim = 256,
                        dropout = 0.1,
                        emb_dropout = 0.1) 
        
    elif param_json_data['parameters']['model'] == "simplevit":
        from models.simplevit import SimpleViT
        generator = SimpleViT(image_size = IMAGE_SIZE,
                              patch_size = 7,
                              num_classes = param_json_data['gene_num'],
                              dim = 512,
                              depth = 6,
                              heads = 8,
                              mlp_dim = 512)
    
    elif param_json_data['parameters']['model'] == 'cait':
        from models.cait import CaiT
        generator = CaiT(
            image_size = IMAGE_SIZE,
            patch_size = 7,
            num_classes = param_json_data['gene_num'],
            dim = 512,
            depth = 6,   # depth of transformer for patch to patch attention only
            cls_depth=2, # depth of cross attention of CLS tokens to patch
            heads = 8,
            mlp_dim = 512,
            dropout = 0.1,
            emb_dropout = 0.1,
            layer_dropout = 0.05)
        
    elif param_json_data['parameters']['model'] == 'cait_small':
        from models.cait import CaiT
        generator = CaiT(
            image_size = IMAGE_SIZE,
            patch_size = 7,
            num_classes = param_json_data['gene_num'],
            dim = 512,
            depth = 6,   # depth of transformer for patch to patch attention only
            cls_depth=2, # depth of cross attention of CLS tokens to patch
            heads = 6,
            mlp_dim = 256,
            dropout = 0.1,
            emb_dropout = 0.1,
            layer_dropout = 0.05)
        
    elif param_json_data['parameters']['model'] == 'resnet18':
        from models.resnet import ResNet18
        generator = ResNet18(num_classes=param_json_data['gene_num'])
        
    elif param_json_data['parameters']['model'] == 'resnet34':
        from models.resnet import ResNet34
        generator = ResNet34(param_json_data['gene_num'])
        
    elif param_json_data['parameters']['model'] == 'resnet50':
        from models.resnet import ResNet50
        generator = ResNet50(param_json_data['gene_num'])
        
    elif param_json_data['parameters']['model'] == 'resnet101':
        from models.resnet import ResNet101
        generator = ResNet101(param_json_data['gene_num'])
        
    elif param_json_data['parameters']['model'] == 'resnet152':
        from models.resnet import ResNet152
        generator = ResNet152(param_json_data['gene_num'])
        
    elif param_json_data['parameters']['model'] == 'vgg11':
        from models.vgg import VGG
        generator = VGG('VGG11', num_classes=param_json_data['gene_num'])
        
    elif param_json_data['parameters']['model'] == 'vgg13':
        from models.vgg import VGG
        generator = VGG('VGG13', num_classes=param_json_data['gene_num'])
        
    elif param_json_data['parameters']['model'] == 'vgg16':
        from models.vgg import VGG
        generator = VGG('VGG16', num_classes=param_json_data['gene_num'])
        
    elif param_json_data['parameters']['model'] == 'vgg19':
        from models.vgg import VGG
        generator = VGG('VGG19', num_classes=param_json_data['gene_num'])
                
    elif param_json_data['parameters']['model'] == 'convmixer':
        from models.convmixer import ConvMixer
        generator = ConvMixer(256, 16, kernel_size=8, patch_size=1, n_classes=param_json_data['gene_num'])
        
    elif param_json_data['parameters']['model'] == 'mlpmixer':
        from models.mlpmixer import MLPMixer
        generator = MLPMixer(
            image_size = IMAGE_SIZE,
            channels = 3,
            patch_size = 7,
            dim = 512,
            depth = 6,
            num_classes = param_json_data['gene_num']
        )

    elif param_json_data['parameters']['model'] == 'swin':   
        from models.swin import swin_l
        generator = swin_l(window_size=7,
                num_classes=param_json_data['gene_num'],
                downscaling_factors=(2,2,2,1))
    else:
        raise Exception("The chosen model is not supported")
    
        
    generator.load_state_dict(model['generator_model_state_dict'])
    discriminator = Discriminator(gene_num=param_json_data['gene_num'], subtype_num=param_json_data['subtype_num'])

    
    discriminator.load_state_dict(model['classifier_model_state_dict'])
    
    generator.eval()
    discriminator.eval()
    
    if cuda:
        generator.cuda()
        discriminator.cuda()
    
    # Tensor types
    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor
    
    with open(opt.output_file, 'w', newline='') as f:
        csvWriter = csv.writer(f)
        csvWriter.writerow(['Object ID','subtype']+param_json_data['geneIDs'])
    
        for i, (imgs, uuid,) in enumerate(tqdm(testDataLoader)):
            batch_size = imgs.shape[0]
            
            # Configure input
            real_imgs = Variable(imgs.type(FloatTensor))
            
            # Generate a batch of images
            gen_trns = generator(real_imgs)
            
            _,_,label = discriminator(gen_trns)
            pred = np.argmax(label.data.cpu().numpy(), axis=1)
        
            # Compute Spearmans Corr
            # spearmanAry = np.asarray([spearmanr(gen_trns.detach().cpu().numpy()[j], trns.detach().cpu().numpy()[j]) for j in range(batch_size)])
            # spearmanAryList.append(spearmanAry)
    
            # Ready to save
            gen_trns = gen_trns.data.cpu().numpy()
            gen_trns *= param_json_data['trns_count_per_cell']
            gen_trns = np.log1p(gen_trns)
            
            for j in range(batch_size):
                row = [uuid[j], pred[j].astype(np.int32)]+gen_trns[j].tolist()
                csvWriter.writerow(row)
                
        # spearmanMean = np.concatenate(spearmanAryList).mean(axis=0)
        # spearmanStddev = np.concatenate(spearmanAryList).std(axis=0)
        # print('Done! Correlation Mean: {:.3f}, Stddev: {:.3f}, p-value Mean: {:.3f}, Stddev: {:.3f}'.format(spearmanMean[0], spearmanStddev[0], spearmanMean[1], spearmanStddev[1]))
