options={
	'locations':49,  # 7*7
	'featureMaps':1024,
	'hdim':1024,  #100
	# 'cores':16,
	'fdim':512,
	'dataset':'UCF11',# ucf11,hmdb51
	'fps':30.0,
	'actions':11,  
	'use_dropout':True,
	'use_wta':False,
	'decay_c':1e-6,
	'last_n':30, # all
	'finetune':False,
	'lrate':1e-3*0.5,
	'momentum':0.8,
	'max_epochs':100,
	'load':True,
	'loadHis':False,

	'model_dir':'params/UCF11/',
	'loadfrom':'model_best-2.npz',
	'saveto':'model-2.npz',
	'bestsaveto':'model_best-2.npz',
	'result':'model_result-2.txt'
}


# options={
# 	'locations':49,  # 7*7
# 	'featureMaps':1024,
# 	'hdim':1024,
# 	'fdim':512,
# 	'dataset':'HMDB51',# ucf11,hmdb51
# 	'fps':30.0,
# 	'actions':51,  
# 	'use_dropout':True,
# 	'use_wta':False,
# 	'decay_c':1e-6,
# 	'last_n':30, # all
# 	'finetune':False,
# 	'lrate':1e-3*0.5,
# 	'momentum':0.8,
# 	'max_epochs':100,
# 	'load':False,
# 	'loadHis':False,

# 	'model_dir':'params/HMDB51/',
# 	'loadfrom':'model_best.npz',
# 	'saveto':'model.npz',
# 	'bestsaveto':'model_best.npz',
# 	'result':'model_result.txt'
# }




"""model name controlls which model you will use"""
# model_name='avg_fnn';
# model_name='avg_lstm';
# model_name='avg_multiChannel';
# model_name='saliency_multiChannel';
# model_name='saliencyLSTM_multiChannel';
model_name='saliencyFgbg_multiChannel';

# model_name='avg_mLSTM';
# model_name='saliency_mLSTM';



options['loadfrom']=options['model_dir']+model_name+'/'+options['loadfrom'];
options['saveto']=options['model_dir']+model_name+'/'+options['saveto'];
options['bestsaveto']=options['model_dir']+model_name+'/'+options['bestsaveto'];
options['result']=options['model_dir']+model_name+'/'+options['result'];




"""import Model"""
module=__import__('model.'+model_name);
model=getattr(module,model_name);
init_params=model.init_params;
build_model=model.build_model;

# from model.avg_fnn import init_params,build_model
# from model.avg_lstm import init_params,build_model
# from model.avg_multiChannel import init_params,build_model
# from model.saliency_multiChannel import init_params,build_model
# from model.saliencyLSTM_multiChannel import init_params,build_model
# from model.saliencyFgbg_multiChannel import init_params,build_model

# from model.lstm_stack import init_params,build_model
# from model.saliency_lstm import init_params,build_model
# from model.avg_linger_lstm import init_params,build_model
# from model.R2RNN import init_params,build_model
# from model.tmp import init_params,build_model


"""import entry"""
from model.entry.entry_withCnnFeature import Train,Test,Visual

"""import datahandler"""
from data.data_handler2 import DataHandler



"""Train || Test"""
# Train(options,init_params,build_model,DataHandler);
# Test(options,init_params,build_model,DataHandler);
Visual(options,init_params,build_model,DataHandler,index=0);