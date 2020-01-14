import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
from scipy.ndimage import imread
from scipy.misc import imsave
import numpy as np
from tensorboardX import SummaryWriter
import os, random
import cv2
import torch.nn.functional as F
from models import pwc_5x5_sigmoid_bilinear
import time
# import torch_msssim
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
# checkpoint = torch.load('/media/machlearn/9A80586280584743/cm/PWC-RefSR/best_model/bicubic_8x_0_pwc5_477675_0.00718599_dict.pkl')
# checkpoint = torch.load('/media/machlearn/9A80586280584743/cm/PWC-RefSR/best_model/bicubic_4x_0_pwc5_431757_0.00536391_dict.pkl')
# checkpoint = torch.load('/media/machlearn/9A80586280584743/cm/PWC-RefSR/best_model/8x_0_dict.pkl')
# checkpoint = torch.load('/media/machlearn/9A80586280584743/cm/PWC-RefSR/best_model/8x_01_dict.pkl')
# checkpoint = torch.load('/media/machlearn/9A80586280584743/cm/PWC-RefSR/best_model/noise01_001_best_dict.pkl')
# checkpoint = torch.load('/media/machlearn/9A80586280584743/cm/PWC-RefSR/best_model/4x_0_pwc0_194002_0.00542384_dict.pkl')
# checkpoint = torch.load('/home/cm/PWC-RefSR/qp32_old/best_dict.pkl')
# checkpoint = torch.load('/home/cm/PWC-RefSR/compression_model/qp22-qp22_best_model.pkl')
# checkpoint = torch.load('/home/cm/PWC-RefSR/compression_model/qp27-qp27_best_model.pkl')
# checkpoint = torch.load('/home/cm/PWC-RefSR/qp27/pwc14_126044_0.01888735.pkl')
# checkpoint = torch.load('/home/cm/PWC-RefSR/compression_model/qp37_gt/pwc0_all_0.02250716.pkl')
# checkpoint = torch.load('/home/cm/PWC-RefSR/compression_model/qp32_gt/pwc0_all_0.01906904.pkl')
# checkpoint = torch.load('/home/cm/PWC-RefSR/noise/2xx/1_all_0.0178897637378.pkl')
# checkpoint = torch.load('/home/cm/PWC-RefSR/noise/1xx/4_50521_0.0156764225778.pkl')
# checkpoint = torch.load('/home/cm/PWC-RefSR/noise/05xx/14_62401_0.0147546693349.pkl')
# checkpoint = torch.load('/home/cm/PWC-RefSR/mixed/27_n10/1_32761_0.0181952396349.pkl')
# checkpoint = torch.load('/home/cm/PWC-RefSR/blur/0.5/3_229641_0.0139461615959.pkl')
# checkpoint = torch.load('/home/cm/PWC-RefSR/noise/0/16_109921_0.0124478758761.pkl')
# checkpoint = torch.load('/home/cm/PWC-RefSR/mixed/32_n07/0_14881_0.020619665976.pkl')
# checkpoint = torch.load('/home/cm/PWC-RefSR/compression_model/qp32-qp32_best_model_old.pkl')
checkpoint = torch.load('/home/cm/huangqian/model/fs_31_all_0.02402605_dict.pkl')
# print(checkpoint)
net = pwc_5x5_sigmoid_bilinear.pwc_residual().cuda()
# net = nn.DataParallel(net)
# print(net)
net.load_state_dict(checkpoint)
padding = torch.nn.ReplicationPad2d([0,0,18,18])
# msssim = torch_msssim.MS_SSIM().cuda()

def cal_msssim(gt,result):
	gt = imread(gt).astype(np.float32)/255.
	result = imread(result).astype(np.float32)/255.
	gt = torch.from_numpy(gt.transpose(2,0,1)).unsqueeze(0)
	result = torch.from_numpy(result.transpose(2,0,1)).unsqueeze(0)
	result, gt = Variable(result).cuda(), Variable(gt).cuda()
	msms = msssim.ms_ssim(gt,result)
	print(msms)
	return msms

def warp( x, flo):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow

    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow

    """
    B, C, H, W = x.size()
    # mesh grid
    xx = torch.arange(0, W).view(1,-1).repeat(H,1)
    yy = torch.arange(0, H).view(-1,1).repeat(1,W)
    xx = xx.view(1,1,H,W).repeat(B,1,1,1)
    yy = yy.view(1,1,H,W).repeat(B,1,1,1)
    grid = torch.cat((xx,yy),1).float()

    if x.is_cuda:
        grid = grid.cuda()
    vgrid = Variable(grid) + flo

    # scale grid to [-1,1]
    vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:]/max(W-1,1)-1.0
    vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:]/max(H-1,1)-1.0

    vgrid = vgrid.permute(0,2,3,1)
    output = nn.functional.grid_sample(x, vgrid)
    mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
    mask = nn.functional.grid_sample(mask, vgrid)

   # if W==128:
        # np.save('mask.npy', mask.cpu().data.numpy())
        # np.save('warp.npy', output.cpu().data.numpy())

    mask[mask<0.9999] = 0
    mask[mask>0] = 1

    return output*mask,mask

def test():
	smsm = np.zeros(600)
	ref_path = '/media/machlearn/9A80586280584743/cm/test4k/Bo_h_22/'
	sr_path = '/media/machlearn/9A80586280584743/cm/test4k/Bo_l_37_u4/'
	result_path = '/media/machlearn/9A80586280584743/cm/test4k/Bo_l_37_ref4/'
	gt_path = '/media/machlearn/9A80586280584743/cm/test4k/Bo_gt/'
	# i = 1
	# ref = ref_path + str(i) + '.png'
	# sr = sr_path + str(i*10-9+4) + '.png'
	# result = result_path + str(i*10-9) + '.png'
	# video(ref,sr,result)
	# gt = gt_path + str(i*10-9) + '.png'
	# smsm[i*10-9] = cal_msssim(gt,ref)
	# ref = result
	# sr = sr_path + str(i*10-8+4) + '.png'
	# result = result_path + str(i*10-8) + '.png'
	# video(ref,sr,result)
	# gt = gt_path + str(i*10-8) + '.png'
	# smsm[i*10-8] = cal_msssim(gt,result)
	# ref = result
	# sr = sr_path + str(i*10-7+4) + '.png'
	# result = result_path + str(i*10-7) + '.png'
	# video(ref,sr,result)
	# gt = gt_path + str(i*10-7) + '.png'
	# smsm[i*10-7] = cal_msssim(gt,result)
	# ref = result
	# sr = sr_path + str(i*10-6+4) + '.png'
	# result = result_path + str(i*10-6) + '.png'
	# video(ref,sr,result)
	# gt = gt_path + str(i*10-6) + '.png'
	# smsm[i*10-6] = cal_msssim(gt,result)
	# ref = result
	# sr = sr_path + str(i*10-5+4) + '.png'
	# result = result_path + str(i*10-5) + '.png'
	# video(ref,sr,result)
	# gt = gt_path + str(i*10-5) + '.png'
	# smsm[i*10-5] = cal_msssim(gt,result)
	for i in range(60,61):
		ref = ref_path + str(i) + '.png'
		sr = sr_path + str(i*10-9+4) + '.png'
		result = result_path + str(i*10-9) + '.png'
		video(ref,sr,result)
		# gt = gt_path + str(i*10-9) + '.png'
		# smsm[i*10-9] = cal_msssim(gt,ref)
		ref = result
		sr = sr_path + str(i*10-8+4) + '.png'
		result = result_path + str(i*10-8) + '.png'
		video(ref,sr,result)
		# gt = gt_path + str(i*10-8) + '.png'
		# smsm[i*10-8] = cal_msssim(gt,result)
		ref = result
		sr = sr_path + str(i*10-7+4) + '.png'
		result = result_path + str(i*10-7) + '.png'
		video(ref,sr,result)
		# gt = gt_path + str(i*10-7) + '.png'
		# smsm[i*10-7] = cal_msssim(gt,result)
		ref = result
		sr = sr_path + str(i*10-6+4) + '.png'
		result = result_path + str(i*10-6) + '.png'
		video(ref,sr,result)
		# gt = gt_path + str(i*10-6) + '.png'
		# smsm[i*10-6] = cal_msssim(gt,result)
		ref = result
		sr = sr_path + str(i*10-5+4) + '.png'
		result = result_path + str(i*10-5) + '.png'
		video(ref,sr,result)
		# gt = gt_path + str(i*10-5) + '.png'
		# smsm[i*10-5] = cal_msssim(gt,result)
		ref = result_path + str(i*10-9) + '.png'
		sr = sr_path + str(i*10-10+4) + '.png'
		result = result_path + str(i*10-10) + '.png'
		video(ref,sr,result)
		# gt = gt_path + str(i*10-10) + '.png'
		# smsm[i*10-10] = cal_msssim(gt,result)
		ref = result
		sr = sr_path + str(i*10-11+4) + '.png'
		result = result_path + str(i*10-11) + '.png'
		video(ref,sr,result)
		# gt = gt_path + str(i*10-11) + '.png'
		# smsm[i*10-11] = cal_msssim(gt,result)
		ref = result
		sr = sr_path + str(i*10-12+4) + '.png'
		result = result_path + str(i*10-12) + '.png'
		video(ref,sr,result)
		# gt = gt_path + str(i*10-12) + '.png'
		# smsm[i*10-12] = cal_msssim(gt,result)
		ref = result
		sr = sr_path + str(i*10-13+4) + '.png'
		result = result_path + str(i*10-13) + '.png'
		video(ref,sr,result)
		# gt = gt_path + str(i*10-13) + '.png'
		# smsm[i*10-13] = cal_msssim(gt,result)
		ref = result
		sr = sr_path + str(i*10-14+4) + '.png'
		result = result_path + str(i*10-14) + '.png'
		video(ref,sr,result)
		# gt = gt_path + str(i*10-14) + '.png'
		# smsm[i*10-14] = cal_msssim(gt,result)
	# np.savetxt('hqp22_lqp37.out',smsm)
	# np.savetxt('hqp22_lqp37_mean.out',np.mean(smsm))
	# print('msssim:{}'.format(np.mean(smsm)))

def single():
	# ref = '/home/cm/iphone/5852/369_mesh.png'
	# sr = '/home/cm/iphone/5852/2981_u3.png'
	# result = '/home/cm/iphone/5852/27_n10.png'
	# video(ref,sr,result)
	ref = '/home/cm/huangqian/realdata/75_cv2_l2.jpg'
	sr = '/home/cm/huangqian/realdata/75_cv1_l2.jpg'
	result = '/home/cm/huangqian/realdata/cv_l2.jpg'
	test_single(ref,sr,result)
	# ref = '/home/cm/exposure/icme/5ms/5hc.png'
	# sr = '/home/cm/exposure/icme/5ms/gn2_1_u2.png'
	# result = '/home/cm/exposure/icme/5ms/gn2_2.png'
	# test_single(ref,sr,result)
	# ref = '/home/cm/huangqian/realdata/75_cv1.jpg'
	# sr = '/home/cm/huangqian/realdata/75_cv2.jpg'
	# result = '/home/cm/huangqian/realdata/cvx.jpg'
	# video(ref,sr,result)
    

def recurrent():
    # ref_path = '/home/cm/PWC-RefSR/Pro_Ma/720p_30fps/fourpeople/i10_qp32/'     # modify
    # sr_path = '/home/cm/PWC-RefSR/Pro_Ma/720p_30fps/fourpeople/l4_qp32_U4/'
    # result_path = '/home/cm/PWC-RefSR/Pro_Ma/720p_30fps/fourpeople/32/'
    # i = 1               
    # ref = ref_path + str(i) + '.png'
    # for j in range(2,7):	
    #     sr = sr_path + str(j) + '.png'
    #     result = result_path + str(j) + '.png'
    #     test_single(ref,sr,result)
    # for i in range(2,300/10+1):
    #     ref = ref_path + str(i) + '.png'
    #     for j in range((i-1)*10-5+2,(i-1)*10+1):
    #         sr = sr_path + str(j) + '.png'
    #         result = result_path + str(j) + '.png'
    #         test_single(ref,sr,result)
    #     for j in range((i-1)*10+2,(i-1)*10+5+2):
    #         sr = sr_path + str(j) + '.png'
    #         result = result_path + str(j) + '.png'
    #         test_single(ref,sr,result)
    # i = 30
    # ref = ref_path + str(i) + '.png'
    # for j in range(297,301):
    #     sr = sr_path + str(j) + '.png'
    #     result = result_path + str(j) + '.png'
    #     test_single(ref,sr,result)
    # ref_path = '/home/cm/PWC-RefSR/Pro_Ma/720p_30fps/krisandsara/i10_qp32/'     # modify
    # sr_path = '/home/cm/PWC-RefSR/Pro_Ma/720p_30fps/krisandsara/l4_qp32_U4/'
    # result_path = '/home/cm/PWC-RefSR/Pro_Ma/720p_30fps/krisandsara/32/'
    # i = 1               
    # ref = ref_path + str(i) + '.png'
    # for j in range(2,7):	
    #     sr = sr_path + str(j) + '.png'
    #     result = result_path + str(j) + '.png'
    #     test_single(ref,sr,result)
    # for i in range(2,300/10+1):
    #     ref = ref_path + str(i) + '.png'
    #     for j in range((i-1)*10-5+2,(i-1)*10+1):
    #         sr = sr_path + str(j) + '.png'
    #         result = result_path + str(j) + '.png'
    #         test_single(ref,sr,result)
    #     for j in range((i-1)*10+2,(i-1)*10+5+2):
    #         sr = sr_path + str(j) + '.png'
    #         result = result_path + str(j) + '.png'
    #         test_single(ref,sr,result)
    # i = 30
    # ref = ref_path + str(i) + '.png'
    # for j in range(297,301):
    #     sr = sr_path + str(j) + '.png'
    #     result = result_path + str(j) + '.png'
    #     test_single(ref,sr,result)

    ref_path = '/home/cm/PWC-RefSR/Pro_Ma/720p_30fps/parkscene/i10_qp32/'     # modify
    sr_path = '/home/cm/PWC-RefSR/Pro_Ma/720p_30fps/parkscene/l4_qp32_U4/'
    result_path = '/home/cm/PWC-RefSR/Pro_Ma/720p_30fps/parkscene/32/'
    # i = 1               
    # ref = ref_path + str(i) + '.png'
    # for j in range(2,7):	
    #     sr = sr_path + str(j) + '.png'
    #     result = result_path + str(j) + '.png'
    #     test_single(ref,sr,result)
    # for i in range(2,100/10+1):
    #     ref = ref_path + str(i) + '.png'
    #     for j in range((i-1)*10-5+2,(i-1)*10+1):
    #         sr = sr_path + str(j) + '.png'
    #         result = result_path + str(j) + '.png'
    #         test_single(ref,sr,result)
    #     for j in range((i-1)*10+2,(i-1)*10+5+2):
    #         sr = sr_path + str(j) + '.png'
    #         result = result_path + str(j) + '.png'
    #         test_single(ref,sr,result)
    i = 10
    ref = ref_path + str(i) + '.png'
    for j in range(97,101):
        sr = sr_path + str(j) + '.png'
        result = result_path + str(j) + '.png'
        test_single(ref,sr,result)

    ref_path = '/home/cm/PWC-RefSR/Pro_Ma/720p_30fps/runner/i10_qp32/'     # modify
    sr_path = '/home/cm/PWC-RefSR/Pro_Ma/720p_30fps/runner/l4_qp32_U4/'
    result_path = '/home/cm/PWC-RefSR/Pro_Ma/720p_30fps/runner/32/'
    i = 1               
    ref = ref_path + str(i) + '.png'
    for j in range(2,7):	
        sr = sr_path + str(j) + '.png'
        result = result_path + str(j) + '.png'
        test_single(ref,sr,result)
    for i in range(2,300/10+1):
        ref = ref_path + str(i) + '.png'
        for j in range((i-1)*10-5+2,(i-1)*10+1):
            sr = sr_path + str(j) + '.png'
            result = result_path + str(j) + '.png'
            test_single(ref,sr,result)
        for j in range((i-1)*10+2,(i-1)*10+5+2):
            sr = sr_path + str(j) + '.png'
            result = result_path + str(j) + '.png'
            test_single(ref,sr,result)
    i = 30
    ref = ref_path + str(i) + '.png'
    for j in range(297,301):
        sr = sr_path + str(j) + '.png'
        result = result_path + str(j) + '.png'
        test_single(ref,sr,result)

def video(ref,sr,result):
	ref = imread(ref).astype(np.float32)/255. #.transpose(1,0,2)
	sr = imread(sr).astype(np.float32)/255.
	ref = torch.from_numpy(ref.transpose(2,0,1)).unsqueeze(0)
	sr = torch.from_numpy(sr.transpose(2,0,1)).unsqueeze(0)
	sr, ref = Variable(sr).cuda(), Variable(ref).cuda()
	sr, ref = padding(sr), padding(ref)
	[b,c,h,w] = ref.size()
	# s = time.time()
	flow = net.FlowNet(torch.cat((sr,ref),1))
	flow = F.upsample(flow,scale_factor=4,mode='bilinear',align_corners=False)*20
	warp_ref,mask_ref = warp(ref,flow.contiguous())
	ref_structure = torch.zeros([b,c,5*5,h,w]).cuda()
	ref_padding = net.pad(warp_ref)
	for i in range(5):
		for j in range(5):
			ref_structure[:,:,i*5+j,:,:] = ref_padding[:,:,i:i+h,j:j+w]
	warp_ref_stru = ref_structure.view(b,c*5*5,h,w)
	del ref, ref_structure, ref_padding
	print('yes')
	features = torch.cat((warp_ref[:,:,:,:1280+64],sr[:,:,:,:1280+64],flow[:,:,:,:1280+64],(warp_ref[:,:,:,:1280+64]-sr[:,:,:,:1280+64])),1)
	mask = net.mask(features.detach())
	mask_sigmoid = F.sigmoid(mask[:,25,:,:])*mask_ref[:,:,:,:1280+64][:,0,:,:]
	ref_r = torch.sum(warp_ref_stru[:,:,:,:1280+64][:,0:25,:,:]*mask[:,0:25,:,:],1)*mask_sigmoid
	ref_g = torch.sum(warp_ref_stru[:,:,:,:1280+64][:,25:50,:,:]*mask[:,0:25,:,:],1)*mask_sigmoid
	ref_b = torch.sum(warp_ref_stru[:,:,:,:1280+64][:,50:75,:,:]*mask[:,0:25,:,:],1)*mask_sigmoid
	ref_contribution = torch.stack([ref_r,ref_g,ref_b],1)
	sr_contribution = ( 1- mask_sigmoid ).unsqueeze(1)
	output1 = ref_contribution + sr[:,:,:,:1280+64]*sr_contribution
	output1x = (output1*255.0).clamp(0,255.0).detach().cpu().numpy().transpose(0,2,3,1).astype(np.uint8).astype(np.float32)/255.
	# imsave(result+'1.png',(output1[0]*255).astype(np.uint8))
	del output1,features,ref_r,ref_g,ref_b,mask_sigmoid,mask,ref_contribution,sr_contribution
	features = torch.cat((warp_ref[:,:,:,1280-32:2560+32],sr[:,:,:,1280-32:2560+32],flow[:,:,:,1280-32:2560+32],(warp_ref[:,:,:,1280-32:2560+32]-sr[:,:,:,1280-32:2560+32])),1)
	mask = net.mask(features.detach())
	mask_sigmoid = F.sigmoid(mask[:,25,:,:])*mask_ref[:,:,:,1280-32:2560+32][:,0,:,:]
	ref_r = torch.sum(warp_ref_stru[:,:,:,1280-32:2560+32][:,0:25,:,:]*mask[:,0:25,:,:],1)*mask_sigmoid
	ref_g = torch.sum(warp_ref_stru[:,:,:,1280-32:2560+32][:,25:50,:,:]*mask[:,0:25,:,:],1)*mask_sigmoid
	ref_b = torch.sum(warp_ref_stru[:,:,:,1280-32:2560+32][:,50:75,:,:]*mask[:,0:25,:,:],1)*mask_sigmoid
	ref_contribution = torch.stack([ref_r,ref_g,ref_b],1)
	sr_contribution = ( 1- mask_sigmoid ).unsqueeze(1)
	output2 = ref_contribution + sr[:,:,:,1280-32:2560+32]*sr_contribution
	output2x = (output2*255.0).clamp(0,255.0).detach().cpu().numpy().transpose(0,2,3,1).astype(np.uint8).astype(np.float32)/255.
	# imsave(result+'2.png',(output2[0]*255).astype(np.uint8))
	del output2,features,ref_r,ref_g,ref_b,mask_sigmoid,mask,ref_contribution,sr_contribution
	features = torch.cat((warp_ref[:,:,:,2560-64:],sr[:,:,:,2560-64:],flow[:,:,:,2560-64:],(warp_ref[:,:,:,2560-64:]-sr[:,:,:,2560-64:])),1)
	mask = net.mask(features.detach())
	mask_sigmoid = F.sigmoid(mask[:,25,:,:])*mask_ref[:,:,:,2560-64:][:,0,:,:]
	ref_r = torch.sum(warp_ref_stru[:,:,:,2560-64:][:,0:25,:,:]*mask[:,0:25,:,:],1)*mask_sigmoid
	ref_g = torch.sum(warp_ref_stru[:,:,:,2560-64:][:,25:50,:,:]*mask[:,0:25,:,:],1)*mask_sigmoid
	ref_b = torch.sum(warp_ref_stru[:,:,:,2560-64:][:,50:75,:,:]*mask[:,0:25,:,:],1)*mask_sigmoid
	ref_contribution = torch.stack([ref_r,ref_g,ref_b],1)
	sr_contribution = ( 1- mask_sigmoid ).unsqueeze(1)
	output3 = ref_contribution + sr[:,:,:,2560-64:]*sr_contribution
	output3x = (output3*255.0).clamp(0,255.0).detach().cpu().numpy().transpose(0,2,3,1).astype(np.uint8).astype(np.float32)/255.
	del output3,features,ref_r,ref_g,ref_b,mask_sigmoid,mask,ref_contribution,sr_contribution
	del warp_ref,mask_ref,flow,sr,warp_ref_stru
	# imsave(result+'3.png',(output3[0]*255).astype(np.uint8))
	# e = time.time()
	# print(e-s)
	output = np.concatenate((output1x[0,8:-8,:1280,:]*255,output2x[0,8:-8,32:-32,:]*255,output3x[0,8:-8,64:,:]*255),axis=1).astype(np.uint8)
	print(output.shape)
	imsave(result, output)

# def test():
# 	ref = '/home/cm/PWC-RefSR/mantis/hr_s2/00_17.png'
# 	sr = '/home/cm/PWC-RefSR/mantis/RefSR_1/00_17_1u.png'
# 	result = '/home/cm/PWC-RefSR/mantis/RefSR_1/00_17_refsr1.png'
# 	img_sr = imread(sr)[:1088,:1920,:].astype(np.float32)/255.  #  1536
# 	img_ref = imread(ref)[:1088,:1920,:].astype(np.float32)/255.
# 	img_sr = torch.from_numpy(img_sr.transpose(2,0,1)).unsqueeze(0)
# 	img_ref = torch.from_numpy(img_ref.transpose(2,0,1)).unsqueeze(0)
# 	sr, ref = Variable(img_sr).cuda(), Variable(img_ref).cuda()
# 	output,warp,mask = net(ref, sr)
# 	output = (output*255.0).clamp(0,255.0).cpu().numpy().transpose(0,2,3,1).astype(np.uint8).astype(np.float32)/255.
# 	imsave(result,(output[0]*255).astype(np.uint8))

def test_single(ref,sr,result):
	sr = imread(sr).astype(np.float32)/255.
	ref = imread(ref).astype(np.float32)/255.
	sr = torch.from_numpy(sr.transpose(2,0,1)).unsqueeze(0)
	ref = torch.from_numpy(ref.transpose(2,0,1)).unsqueeze(0)
	sr, ref = Variable(sr).cuda(), Variable(ref).cuda()
	sr, ref = padding(sr), padding(ref)
	output,warp,mask = net(ref, sr)
	output = (output*255.0).clamp(0,255.0).cpu().numpy().transpose(0,2,3,1).astype(np.uint8).astype(np.float32)/255.
	imsave(result,(output[0]*255).astype(np.uint8))  #[0,24:744,:,:]

def mantis_all():
	# ref = '/home/cm/PWC-RefSR/mantis/hr_s2/00_17.png'
	# sr = '/home/cm/PWC-RefSR/mantis/RefSR_1/00_17_1u.png'
	# result = '/home/cm/PWC-RefSR/mantis/RefSR_1/00_17_refsr.png'
	# mantis(ref,sr,result)
	for i in range(0,3):
		for j in range(0,12):
			ref = '/home/cm/PWC-RefSR/mantis/new_data/hr_c/' + str(i).zfill(2) + '_' + str(j).zfill(2) + '.png'
			sr = '/home/cm/PWC-RefSR/mantis/new_data/refsr_1u/' + str(i).zfill(2) + '_' + str(j).zfill(2) + '.png'
			result = '/home/cm/PWC-RefSR/mantis/new_data/refsr_2/' + str(i).zfill(2) + '_' + str(j).zfill(2) + '.png'
			mantis(ref,sr,result)
			# test_single(ref,sr,result)


def mantis(ref,sr,result):
	ref = imread(ref).astype(np.float32)/255.
	sr = imread(sr).astype(np.float32)/255.
	ref = torch.from_numpy(ref.transpose(2,0,1)).unsqueeze(0)
	sr = torch.from_numpy(sr.transpose(2,0,1)).unsqueeze(0)
	sr, ref = Variable(sr).cuda(), Variable(ref).cuda()
	[b,c,h,w] = ref.size()
	# s = time.time()
	flow = net.FlowNet(torch.cat((sr,ref),1))
	flow = F.upsample(flow,scale_factor=4,mode='bilinear',align_corners=False)*20
	warp_ref,mask_ref = warp(ref,flow.contiguous())
	ref_structure = torch.zeros([b,c,5*5,h,w]).cuda()
	ref_padding = net.pad(warp_ref)
	for i in range(5):
		for j in range(5):
			ref_structure[:,:,i*5+j,:,:] = ref_padding[:,:,i:i+h,j:j+w]
	warp_ref_stru = ref_structure.view(b,c*5*5,h,w)
	del ref, ref_structure, ref_padding
	features = torch.cat((warp_ref[:,:,:,:1280+64],sr[:,:,:,:1280+64],flow[:,:,:,:1280+64],(warp_ref[:,:,:,:1280+64]-sr[:,:,:,:1280+64])),1)
	mask = net.mask(features.detach())
	mask_sigmoid = F.sigmoid(mask[:,25,:,:])*mask_ref[:,:,:,:1280+64][:,0,:,:]
	ref_r = torch.sum(warp_ref_stru[:,:,:,:1280+64][:,0:25,:,:]*mask[:,0:25,:,:],1)*mask_sigmoid
	ref_g = torch.sum(warp_ref_stru[:,:,:,:1280+64][:,25:50,:,:]*mask[:,0:25,:,:],1)*mask_sigmoid
	ref_b = torch.sum(warp_ref_stru[:,:,:,:1280+64][:,50:75,:,:]*mask[:,0:25,:,:],1)*mask_sigmoid
	ref_contribution = torch.stack([ref_r,ref_g,ref_b],1)
	sr_contribution = ( 1- mask_sigmoid ).unsqueeze(1)
	output1 = ref_contribution + sr[:,:,:,:1280+64]*sr_contribution
	output1 = (output1*255.0).clamp(0,255.0).detach().cpu().numpy().transpose(0,2,3,1).astype(np.uint8).astype(np.float32)/255.
	# imsave(result+'1.png',(output1[0]*255).astype(np.uint8))

	features = torch.cat((warp_ref[:,:,:,1280-32:2560+32],sr[:,:,:,1280-32:2560+32],flow[:,:,:,1280-32:2560+32],(warp_ref[:,:,:,1280-32:2560+32]-sr[:,:,:,1280-32:2560+32])),1)
	mask = net.mask(features.detach())
	mask_sigmoid = F.sigmoid(mask[:,25,:,:])*mask_ref[:,:,:,1280-32:2560+32][:,0,:,:]
	ref_r = torch.sum(warp_ref_stru[:,:,:,1280-32:2560+32][:,0:25,:,:]*mask[:,0:25,:,:],1)*mask_sigmoid
	ref_g = torch.sum(warp_ref_stru[:,:,:,1280-32:2560+32][:,25:50,:,:]*mask[:,0:25,:,:],1)*mask_sigmoid
	ref_b = torch.sum(warp_ref_stru[:,:,:,1280-32:2560+32][:,50:75,:,:]*mask[:,0:25,:,:],1)*mask_sigmoid
	ref_contribution = torch.stack([ref_r,ref_g,ref_b],1)
	sr_contribution = ( 1- mask_sigmoid ).unsqueeze(1)
	output2 = ref_contribution + sr[:,:,:,1280-32:2560+32]*sr_contribution
	output2 = (output2*255.0).clamp(0,255.0).detach().cpu().numpy().transpose(0,2,3,1).astype(np.uint8).astype(np.float32)/255.
	# imsave(result+'2.png',(output2[0]*255).astype(np.uint8))

	features = torch.cat((warp_ref[:,:,:,2560-64:],sr[:,:,:,2560-64:],flow[:,:,:,2560-64:],(warp_ref[:,:,:,2560-64:]-sr[:,:,:,2560-64:])),1)
	mask = net.mask(features.detach())
	mask_sigmoid = F.sigmoid(mask[:,25,:,:])*mask_ref[:,:,:,2560-64:][:,0,:,:]
	ref_r = torch.sum(warp_ref_stru[:,:,:,2560-64:][:,0:25,:,:]*mask[:,0:25,:,:],1)*mask_sigmoid
	ref_g = torch.sum(warp_ref_stru[:,:,:,2560-64:][:,25:50,:,:]*mask[:,0:25,:,:],1)*mask_sigmoid
	ref_b = torch.sum(warp_ref_stru[:,:,:,2560-64:][:,50:75,:,:]*mask[:,0:25,:,:],1)*mask_sigmoid
	ref_contribution = torch.stack([ref_r,ref_g,ref_b],1)
	sr_contribution = ( 1- mask_sigmoid ).unsqueeze(1)
	output3 = ref_contribution + sr[:,:,:,2560-64:]*sr_contribution
	output3 = (output3*255.0).clamp(0,255.0).detach().cpu().numpy().transpose(0,2,3,1).astype(np.uint8).astype(np.float32)/255.
	# imsave(result+'3.png',(output3[0]*255).astype(np.uint8))
	# e = time.time()
	# print(e-s)
	output = np.concatenate((output1[0,8:-8,:1280,:]*255,output2[0,8:-8,32:-32,:]*255,output3[0,8:-8,64:,:]*255),axis=1).astype(np.uint8)
	print(output.shape)
	imsave(result, output)


with torch.no_grad():
	# mantis_all()
	# test()
	single()
	# recurrent()
	# test(net)