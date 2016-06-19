options={
	'videosize':64, 
	'featureMaps':1024,
	'fdim':512,
	'dataset':'UCF11',
	'fps':90.0,
	'actions':11,  
	'use_dropout':True,
	'use_wta':False,
	'decay_c':1e-6,
	'last_n':30, # all
	'finetune':False,
	'lrate':1e-3*0.5,
	'momentum':0.8,
	'max_epochs':100,
	'load':False,
	'loadHis':False,

	'model_dir':'params/UCF11/',
	'loadfrom':'model_best.npz',
	'saveto':'model.npz',
	'bestsaveto':'model_best.npz',
	'result':'model_result.txt'
}


# options={
# 	'videosize':64, 
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


model_name='cnn_fnn';


options['loadfrom']=options['model_dir']+model_name+'/'+options['loadfrom'];
options['saveto']=options['model_dir']+model_name+'/'+options['saveto'];
options['bestsaveto']=options['model_dir']+model_name+'/'+options['bestsaveto'];
options['result']=options['model_dir']+model_name+'/'+options['result'];



"""import Model"""
module=__import__('model.'+model_name);
model=getattr(module,model_name);
init_params=model.init_params;
build_model=model.build_model;


from model.entry.entry_withVideo import Train,Test




"""Data"""
from data.data_handler2 import DataHandler



"""Train || Test"""
Train(options,init_params,build_model,DataHandler);
# Test(options,init_params,build_model,DataHandler);