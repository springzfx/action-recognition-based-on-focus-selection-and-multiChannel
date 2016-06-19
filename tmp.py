import theano
import numpy as np

x=theano.tensor.dvector('x');

x_p=theano.printing.Print(attrs=("shape",))(x);

f=theano.function([x],x_p**2);

f([1,2,3])
