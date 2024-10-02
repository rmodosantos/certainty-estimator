from torch.utils.data import DataLoader


def trainNet(net, data_train, optimizer,criterion, batch_size=4, epochs=4, printr=False, save_loss=False):
 
    """
        Function to train a neural network.
        This function trains a neural network using the specified training dataset and optimizer. 
        It performs training for the specified number of epochs, with the option to print the running loss
        and save it for analysis.
        
        Parameters:
        - net (nn.Module): The neural network to be trained.
        - data_train (torch.utils.data.Dataset): Training dataset.
        - optimizer (torch.optim.Optimizer): The optimizer used for training.
        - criterion (callable): Loss function to compute the training loss.
        - batch_size (int, optional): Size of the mini-batch used in optimization. Defaults to 10.
        - epochs (int, optional): Number of training epochs. Defaults to 4.
        - ACh_loss (bool, optional): Whether to use ACh-weighted loss. Defaults to False.
        - printr (bool, optional): Boolean to determine whether running loss is displayed during execution. Defaults to False.
        - save_loss (bool, optional): Boolean to indicate whether to store information about the running loss. Defaults to False.

        Returns:
        - rloss (list): List of training losses if save_loss is True, otherwise, None.
    
    """
    
    import time
    
    net.train()
    
    # Create data loader object
    loader = DataLoader(data_train,batch_size=batch_size, shuffle=True)

    start = time.time()
    epoch_start = 0
    rloss = []
    
    
    # Go over 1000 training instances (1 epoch) multiple times
    for epoch in range(epochs):  

        epoch_start += 1
        running_loss = 0.0

        for i, (image,target) in enumerate(loader):

            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            #print(image.size())
            # Compute network output
            outputs=net(image)
            
            #print('target',target)
            #print('output',outputs)
            # Calculate loss
            loss = criterion(outputs, target)
           
        
            #update weights based on backpropagation
            loss.backward()
            
            # Clip to avoid exploding gradients
            nn.utils.clip_grad_norm_(net.parameters(), max_norm=1)
            
            # Update optimizer
            optimizer.step()

            # Output
            if save_loss:
                rloss.append(loss.item())

            if printr:
                # Print statistics
                running_loss += loss.item()

                if (i+1) % 10 == 0:    # print every 10 mini-batches
                    print(f"[{epoch+1}, {i+1}] loss: {np.round(running_loss/(10),7)}")

                    running_loss = 0.0

    end = time.time()

    if printr:
        print('Finished Training')
        print('training time ', end-start)

    if save_loss:
        return rloss