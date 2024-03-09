

# python basicsr/train.py -opt options/FLIM3D/FLIM3D_PECBNet_train.yml
# python basicsr/test.py -opt options/FLIM3D/FLIM3D_PECBNet_test.yml


# python basicsr/train.py -opt options/FLIM3D/FLIM3D_PECBNet_downsampling_train.yml
# python basicsr/test.py -opt options/FLIM3D/FLIM3D_PECBNet_downsampling_test.yml


# python basicsr/train.py -opt options/FLIM3D/FLIM3D_PECBNet_sparse_frames_train.yml
# python basicsr/test.py -opt options/FLIM3D/FLIM3D_PECBNet_sparse_frames_test.yml

#python basicsr/train.py -opt options/FLIM3D/FLIM3D_PECBNet_temporal_sparsity_train.yml
#python basicsr/test.py -opt options/FLIM3D/FLIM3D_PECBNet_temporal_sparsity_test.yml



python basicsr/test.py -opt options/FLIM3D/FLIM3D_PECBNet_endoscopy_test.yml

#python basicsr/test.py -opt options/FLIM3D_PECBNet_16_channels_test.yml

##################  stack dataset  #######################
# python basicsr/train.py -opt options/FLIM3D/FLIM3D_PECBNet_stack3D_train.yml
# python basicsr/test.py -opt options/FLIM3D/FLIM3D_PECBNet_stack3D_test.yml

# python basicsr/train.py -opt options/FLIM3D/FLIM3D_PECBNet_stack3D_sparse_frames_train.yml



##################  SRS dataset  #######################
# python basicsr/train.py -opt options/SRS/SRS_PECBNet_train.yml
#python basicsr/test.py -opt options/SRS/SRS_PECBNet_test.yml