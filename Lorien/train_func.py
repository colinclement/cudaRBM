def train_rbm(sample, learning_rate=0.1, training_epochs=15, batch_size=20, n_hidden = 5, persistent=True,
              output_folder='rbm_saves/', filename='spin_glass_rbm_no_theano.pklz'):
    """
    Initiate an RBM class and train on spin glass data
    inputs: sample - samples from a spin glass
            learning_rate - Size of gradient descent steps during training
            training_epochs - Number of times all of the training vectors are used once to update the weights
            batch_size - Number of samples in each training vector batch
            n_hidden - dimension of the top layer
            output_folder - folder in which to save the rbm
            filename - filename under which to save the rbm
    outputs:
            saves RBM after training and returns it
    """

    #Initiate the data
    #train_set_x = numpy.asarray(sample[:6000])
    #test_set_x = numpy.asarray(sample[6000:])
    train_set_x = sample
    n_train_batches = train_set_x.shape[0] / batch_size
    n_visible=train_set_x.shape[1]
    #initiate the RBM
    input = train_set_x[0:batch_size]
    rbm = RBM(input=input, n_visible=train_set_x.shape[1], n_hidden=n_hidden)
    #get starting cost and updates
    cost = rbm.get_cost_updates(n_visible=n_visible, n_hidden=n_hidden, lr=learning_rate, persistent=persistent, k=15)
    #makes the directory for saves
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    
    #### TRAIN THE RBM ####
    
    start_time = time.time()
  
    # go through training epochs
    for epoch in xrange(training_epochs):
        
        # go through the training set
        mean_cost = []
        for batch_index in xrange(n_train_batches):
            rbm.input = train_set_x[batch_index * batch_size: (batch_index + 1) * batch_size]
            cost = rbm.get_cost_updates(n_visible=n_visible, n_hidden=n_hidden, lr=learning_rate, persistent=persistent, k=15)
            mean_cost.append(cost)
        print 'Training epoch %d, cost is ' % epoch, numpy.mean(mean_cost)
        
    end_time = time.time()
    
    pretraining_time = (end_time - start_time)
    print ('Training took %f minutes' % (pretraining_time / 60.))
    
    # save the RBM
    fn = output_folder+filename
    with gzip.open(fn,'wb') as f:
        pickle.dump(rbm,f,-1)
    return rb
