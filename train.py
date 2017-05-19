import tensorflow as tf
import numpy
import glob

def read_and_decode_tfrecord(filename,shape):
	filename_queue=tf.train.string_input_producer([filename],num_epochs=None)
	reader=tf.TFRecordReader()
	_,serialized_example=reader.read(filename_qeueu)
	features=tf.parse_single_example(serialized_example,
		features={
			'height':tf.FixedLenFeature([],tf.int64),
			'width':tf.FixedLenFeature([],tf.int64),
			'input_data_raw':tf.FixedLenFeature([],dtype=tf.string)
			'gt_data_raw':tf.FixedLenFeature([],dtype=tf.string)
		})
	image=tf.decode_raw(features['input_data_raw'],tf.float32)
	gt=tf.decode_raw(features['gt_data_raw'],tf.float32)
	image_shape=tf.pack(shape)
	input_image=tf.reahspe(image,image_shape)
	gt_image=tf.reshape(gt,image_shape)
	input_image_batch,gt_image_batch=tf.train.batch([input_image,gt_image],batch_size=batch_size)
	return input_image_batch,gt_image_batch

def train():
	pass

def initialize_weights(nL):
	
	pass
def relu(x):
	return tf.nn.relu(x)
def conv3d(x,W,b,stride=1,padding='SAME'):
	x=tf.nn.conv3d(x,W,strides=[1,stride,stride,stride,1],padding=padding)
	return tf.nn.bias_ass(x,b)

def maxpool3d(x,k):
	return tf.nn.max_pool3d(x,k=[1,k,k,k,1],strides=[1,k,k,k,1],padding='SAME')
def avgpool3d(x,k):
	return tf.nn.avg_pool3d(x,k=[1,k,k,k,1],strides=[1,k,k,k,1],padding='SAME')
def deconv3d(x,W,output_shape,stride=1):
	return tv.nn.conv3d_transpose(x,W,output_shape=output_shape,strides=[1,stride,stride,stride,1])

path=''
data_shape=[]
data=load_data(path,data_shape)


learning_rate=0.001
training_iter=0
batch_size=100
display_step=100

n_input_h=0 # shape of an input data
n_input_v=0
n_input_c=0
n_output_h=n_input_h # shape of an output dat
n_output_v=n_input_v
n_output_c=n_input_c

x=tf.placeholder(tf.float32,[None,n_input_c*n_input_h*n_input_v]) # check order of dimension
y=tf.placeholder(tf.float32,[None,n_input_c*n_input_h*n_input_v])
keep_prob=tf.placeholder(tf.float32)

# filter [filter_depth,filter_height, filter_width,in_channel,out_channel
# input: [batch,in_depth,in_height,in_width,in_channel] 
weights={
	# size of filter may vary for channel, width, and height
	'wc1':tf.Variable(tf.random_normal([5,5,5,1,32])),
	'wc2':tf.Variable(tf.random_normal([5,5,5,32,32])),
	'wc3':tf.Variable(tf.random_normal([5,5,5,32,32])),
	'wc4':tf.Variable(tf.random_normal([5,5,5,32,32])),
	'wc5':tf.Variable(tf.random_normal([5,5,5,32,1]))
}
biases={
	'bc1':tf.Variable(tf.random_normal([32])),
	'bc2':tf.Variable(tf.random_normal([32])),
	'bc3':tf.Variable(tf.random_normal([32])),
	'bc4':tf.Variable(tf.random_normal([32])),
	'bc5':tf.Variable(tf.random_normal([1]))
}
input_data=load_data("../../ConeBeamPatch/SinoPatch/170/0000/")
def conv_net(x,weights,biases,dropout):
	x=tf.reshape(x,[n_input_c,n_input_h,n_input_v])
	conv1=relu(conv3d(x,weights['wc1'],biases['bc1']))
	conv2=relu(conv2d(conv1,weights['wc2'],biases['bc2']),padding='VALID')
	conv3=relu(conv3d(conv2,weights['wc3'],biases['bc3']),padding='VALID')
	conv4=deconv3d(conv3,weights['wc4'],biases['bc4'])
	out=conv3d(conv4,weights['wc5'],biases['bc5'])
	return out
def get_next_batch():
	pass
output=conv_net(x,weights,biases,keep_prob)
cost=tf.reduce_mean(tf.nn.l2_loss(output-y))
optimizer=tf.train.AdapOptimizer(learning_rate=learning_rate).minimize(cost)

init=tf.global_variables_initializer()

# deconvolution can be performed using conv2d_transpose?
with tf.Session() as sess:
	sess.run(init)
	step=1
	while step<tarining_iters:
		batch_x,batch_y=get_next_batch(batch_size)
		# how to make get_next_batch function
		sess.run(optimizer,feed_dict={x:batch_x,y:batch_y,keep_prob:dropout})
		if step% display_step==0:
			## how to separate training batch and testing batch?
			loss=sess.run(cost,feed_dict={x:batch_x_test,y: batch_y_test,keep_prob:1})
			print "Iter " + str(step)+ ", Minibatch loss= "+"{:.6f}".format(loss)
		step+=1
	print "Optimization done"	
