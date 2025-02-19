from ComputationalGraphPrimer import *

class myCGP_sgdp(ComputationalGraphPrimer):

    def __init__(self, *args, **kwargs ):
        super().__init__(*args, **kwargs)

    def run_training_loop_one_neuron_model(self, training_data, momentum_coeff=0.9):
        self.vals_for_learnable_params = {param: random.uniform(0,1) for param in self.learnable_params}
        self.bias = random.uniform(0,1)  

         

        velocity = [0] * (len(self.learnable_params) + 1)       

         

        class DataLoader:
            def __init__(self, training_data, batch_size):
                self.training_data = training_data
                self.batch_size = batch_size
                self.class_0_samples = [(item, 0) for item in self.training_data[0]]   
                self.class_1_samples = [(item, 1) for item in self.training_data[1]]   

            def __len__(self):
                return len(self.training_data[0]) + len(self.training_data[1])

            def _getitem(self):    
                cointoss = random.choice([0,1])                           
                if cointoss == 0:
                    return random.choice(self.class_0_samples)
                else:
                    return random.choice(self.class_1_samples)            

            def getbatch(self):
                batch_data,batch_labels = [],[]                            
                maxval = 0.0                                               
                for _ in range(self.batch_size):
                    item = self._getitem()
                    if np.max(item[0]) > maxval: 
                        maxval = np.max(item[0])
                    batch_data.append(item[0])
                    batch_labels.append(item[1])
                batch_data = [item/maxval for item in batch_data]          
                batch = [batch_data, batch_labels]
                return batch                

        data_loader = DataLoader(training_data, batch_size=self.batch_size)
        loss_running_record = []
        i = 0
        avg_loss_over_iterations = 0.0                                    

        for i in range(self.training_iterations):
            data = data_loader.getbatch()
            data_tuples_in_batch = data[0]
            class_labels_in_batch = data[1]
            y_preds, deriv_sigmoids =  self.forward_prop_one_neuron_model(data_tuples_in_batch)     
            loss = sum([(abs(class_labels_in_batch[i] - y_preds[i]))**2 for i in range(len(class_labels_in_batch))])  
            avg_loss_over_iterations += loss / float(len(class_labels_in_batch))
            if i%(self.display_loss_how_often) == 0: 
                avg_loss_over_iterations /= self.display_loss_how_often
                loss_running_record.append(avg_loss_over_iterations)
                # print("[iter=%d]  loss = %.4f" %  (i+1, avg_loss_over_iterations))                 
                avg_loss_over_iterations = 0.0                                                     
            y_errors_in_batch = list(map(operator.sub, class_labels_in_batch, y_preds))

         

            velocity = self.backprop_and_update_params_one_neuron_model(data_tuples_in_batch, y_preds, y_errors_in_batch, deriv_sigmoids, velocity, momentum_coeff)    

        return loss_running_record
         


    def backprop_and_update_params_one_neuron_model(self, data_tuples_in_batch, predictions, y_errors_in_batch, deriv_sigmoids, velocity, momentum_coeff):
        input_vars = self.independent_vars
        input_vars_to_param_map = self.var_to_var_param[self.output_vars[0]]                  
        param_to_vars_map = {param : var for var, param in input_vars_to_param_map.items()}   
        vals_for_learnable_params = self.vals_for_learnable_params
        for i,param in enumerate(self.vals_for_learnable_params):
            
            partial_of_loss_wrt_param = 0.0
            for j in range(self.batch_size):
                vals_for_input_vars_dict =  dict(zip(input_vars, list(data_tuples_in_batch[j])))
                partial_of_loss_wrt_param   +=   -  y_errors_in_batch[j] * vals_for_input_vars_dict[param_to_vars_map[param]] * deriv_sigmoids[j]
            partial_of_loss_wrt_param /=  float(self.batch_size)

             

            velocity[i] = momentum_coeff * velocity[i] + partial_of_loss_wrt_param
            step = self.learning_rate * velocity[i] * -1 

             
            self.vals_for_learnable_params[param] += step

        y_error_avg = sum(y_errors_in_batch) / float(self.batch_size)
        deriv_sigmoid_avg = sum(deriv_sigmoids) / float(self.batch_size)

         

        total_partial = y_error_avg * deriv_sigmoid_avg
        velocity[-1] = momentum_coeff * velocity[-1] + total_partial
        b_step = self.learning_rate * velocity[-1]
        self.bias += b_step

        return velocity
         

    def run_training_loop_multi_neuron_model(self, training_data, momentum_coeff):

        class DataLoader:
            def __init__(self, training_data, batch_size):
                self.training_data = training_data
                self.batch_size = batch_size
                self.class_0_samples = [(item, 0) for item in self.training_data[0]]    
                self.class_1_samples = [(item, 1) for item in self.training_data[1]]    

            def __len__(self):
                return len(self.training_data[0]) + len(self.training_data[1])

            def _getitem(self):    
                cointoss = random.choice([0,1])                            
                if cointoss == 0:
                    return random.choice(self.class_0_samples)
                else:
                    return random.choice(self.class_1_samples)            

            def getbatch(self):
                batch_data,batch_labels = [],[]                            
                maxval = 0.0                                               
                for _ in range(self.batch_size):
                    item = self._getitem()
                    if np.max(item[0]) > maxval: 
                        maxval = np.max(item[0])
                    batch_data.append(item[0])
                    batch_labels.append(item[1])
                batch_data = [item/maxval for item in batch_data]              
                batch = [batch_data, batch_labels]
                return batch                
        
         

        velocity_of_param = {param : 0.0 for param in self.all_params}
        bias_vals =   {i : [random.uniform(0,1) for j in range( self.layers_config[i])]  for i in range(1, self.num_layers)}

         

        
        self.vals_for_learnable_params = {param: random.uniform(0,1) for param in self.learnable_params}
        self.bias =   {i : [random.uniform(0,1) for j in range( self.layers_config[i] ) ]  for i in range(1, self.num_layers)}

        data_loader = DataLoader(training_data, batch_size=self.batch_size)
        loss_running_record = []
        i = 0
        avg_loss_over_iterations = 0.0                                          

        for i in range(self.training_iterations):
            data = data_loader.getbatch()
            data_tuples = data[0]
            class_labels = data[1]
            self.forward_prop_multi_neuron_model(data_tuples)                                       
            predicted_labels_for_batch = self.forw_prop_vals_at_layers[self.num_layers-1]           
            y_preds =  [item for sublist in  predicted_labels_for_batch  for item in sublist]       
            loss = sum([(abs(class_labels[i] - y_preds[i]))**2 for i in range(len(class_labels))])  
            loss_avg = loss / float(len(class_labels))                                              
            avg_loss_over_iterations += loss_avg                                                    
            if i%(self.display_loss_how_often) == 0: 
                avg_loss_over_iterations /= self.display_loss_how_often
                loss_running_record.append(avg_loss_over_iterations)
                # print("[iter=%d]  loss = %.4f" %  (i+1, avg_loss_over_iterations))                  
                avg_loss_over_iterations = 0.0                                                      
            y_errors_in_batch = list(map(operator.sub, class_labels, y_preds))

             

            self.backprop_and_update_params_multi_neuron_model(y_preds, y_errors_in_batch, velocity_of_param, bias_vals, momentum_coeff)

             

        return loss_running_record
        
    def backprop_and_update_params_multi_neuron_model(self, predictions, y_errors, velocity_of_param, bias_vals, momentum_coeff):
        
        pred_err_backproped_at_layers =   [ {i : [None for j in range( self.layers_config[i] ) ]  
                                                                  for i in range(self.num_layers)} for _ in range(self.batch_size) ]
        
        partial_of_loss_wrt_params = {param : 0.0 for param in self.all_params}
        bias_changes =   {i : [0.0 for j in range( self.layers_config[i] ) ]  for i in range(1, self.num_layers)}

        for b in range(self.batch_size):
            pred_err_backproped_at_layers[b][self.num_layers - 1] = [ y_errors[b] ]
            for back_layer_index in reversed(range(1,self.num_layers)):             
                input_vals = self.forw_prop_vals_at_layers[back_layer_index -1]     
                deriv_sigmoids =  self.gradient_vals_for_layers[back_layer_index]   
                vars_in_layer  =  self.layer_vars[back_layer_index]                 
                vars_in_next_layer_back  =  self.layer_vars[back_layer_index - 1]   
                vals_for_input_vars_dict =  dict(zip(vars_in_next_layer_back, self.forw_prop_vals_at_layers[back_layer_index - 1][b]))   
                
                layer_params = self.layer_params[back_layer_index]         
                transposed_layer_params = list(zip(*layer_params))                  
                for k,var1 in enumerate(vars_in_next_layer_back):
                    for j,var2 in enumerate(vars_in_layer):
                        pred_err_backproped_at_layers[b][back_layer_index - 1][k] = sum([self.vals_for_learnable_params[transposed_layer_params[k][i]]
                                                                                       * pred_err_backproped_at_layers[b][back_layer_index][i]
                                                                                                                  for i in range(len(vars_in_layer))])
                for j,var in enumerate(vars_in_layer):
                    layer_params = self.layer_params[back_layer_index][j]           
                    input_vars_to_param_map = self.var_to_var_param[var]            
                    param_to_vars_map = {param : var for var, param in input_vars_to_param_map.items()}   


                    for i,param in enumerate(layer_params):
                        partial_of_loss_wrt_params[param]   +=   pred_err_backproped_at_layers[b][back_layer_index][j] * \
                                                                        vals_for_input_vars_dict[param_to_vars_map[param]] * deriv_sigmoids[b][j]

                for k,var1 in enumerate(vars_in_next_layer_back):
                    for j,var2 in enumerate(vars_in_layer):
                        if back_layer_index-1 > 0:
                            bias_changes[back_layer_index-1][k] += pred_err_backproped_at_layers[b][back_layer_index - 1][k] * deriv_sigmoids[b][j] 
 
        
        for param in partial_of_loss_wrt_params: 
            partial_of_loss_wrt_param = partial_of_loss_wrt_params[param] /  float(self.batch_size)   

             

            velocity_of_param[param] = momentum_coeff * velocity_of_param[param] + partial_of_loss_wrt_param
            step = self.learning_rate * velocity_of_param[param] 
            self.vals_for_learnable_params[param] += step

             
 
        for layer_index in range(1,self.num_layers):           
            for k in range(self.layers_config[layer_index]):
                 

                bias_vals[layer_index][k] = momentum_coeff * bias_vals[layer_index][k] + ( bias_changes[layer_index][k] / float(self.batch_size) )
                b_step = self.learning_rate * bias_vals[layer_index][k] 
                self.bias[layer_index][k]  += b_step

                 



class myCGP_adam(ComputationalGraphPrimer):

    def __init__(self, *args, **kwargs ):
        super().__init__(*args, **kwargs)

    def run_training_loop_one_neuron_model(self, training_data, b1, b2, eps ):
        self.vals_for_learnable_params = {param: random.uniform(0,1) for param in self.learnable_params}
        self.bias = random.uniform(0,1)       

         

        velocity = [0] * (len(self.learnable_params) + 1)
        moment = [0] * (len(self.learnable_params) + 1)       

         

        class DataLoader:
            def __init__(self, training_data, batch_size):
                self.training_data = training_data
                self.batch_size = batch_size
                self.class_0_samples = [(item, 0) for item in self.training_data[0]]    
                self.class_1_samples = [(item, 1) for item in self.training_data[1]]    

            def __len__(self):
                return len(self.training_data[0]) + len(self.training_data[1])

            def _getitem(self):    
                cointoss = random.choice([0,1])                            
                if cointoss == 0:
                    return random.choice(self.class_0_samples)
                else:
                    return random.choice(self.class_1_samples)            

            def getbatch(self):
                batch_data,batch_labels = [],[]                             
                maxval = 0.0                                                
                for _ in range(self.batch_size):
                    item = self._getitem()
                    if np.max(item[0]) > maxval: 
                        maxval = np.max(item[0])
                    batch_data.append(item[0])
                    batch_labels.append(item[1])
                batch_data = [item/maxval for item in batch_data]           
                batch = [batch_data, batch_labels]
                return batch                


        data_loader = DataLoader(training_data, batch_size=self.batch_size)
        loss_running_record = []
        i = 0
        avg_loss_over_iterations = 0.0                                    

        for i in range(self.training_iterations):
            data = data_loader.getbatch()
            data_tuples_in_batch = data[0]
            class_labels_in_batch = data[1]
            y_preds, deriv_sigmoids =  self.forward_prop_one_neuron_model(data_tuples_in_batch)      
            loss = sum([(abs(class_labels_in_batch[i] - y_preds[i]))**2 for i in range(len(class_labels_in_batch))])   
            avg_loss_over_iterations += loss / float(len(class_labels_in_batch))
            if i%(self.display_loss_how_often) == 0: 
                avg_loss_over_iterations /= self.display_loss_how_often
                loss_running_record.append(avg_loss_over_iterations)
                # print("[iter=%d]  loss = %.4f" %  (i+1, avg_loss_over_iterations))                  
                avg_loss_over_iterations = 0.0                                                      
            y_errors_in_batch = list(map(operator.sub, class_labels_in_batch, y_preds))

         

            self.backprop_and_update_params_one_neuron_model(data_tuples_in_batch, y_preds, y_errors_in_batch, deriv_sigmoids, moment, velocity, b1, b2, eps, i)   

        return loss_running_record
         


    def backprop_and_update_params_one_neuron_model(self, data_tuples_in_batch, predictions, y_errors_in_batch, deriv_sigmoids, moment, velocity, b1, b2, eps, train_itr):
        input_vars = self.independent_vars
        input_vars_to_param_map = self.var_to_var_param[self.output_vars[0]]                   
        param_to_vars_map = {param : var for var, param in input_vars_to_param_map.items()}    
        vals_for_learnable_params = self.vals_for_learnable_params
        for i,param in enumerate(self.vals_for_learnable_params):
             
            partial_of_loss_wrt_param = 0.0
            for j in range(self.batch_size):
                vals_for_input_vars_dict =  dict(zip(input_vars, list(data_tuples_in_batch[j])))
                partial_of_loss_wrt_param   +=   -  y_errors_in_batch[j] * vals_for_input_vars_dict[param_to_vars_map[param]] * deriv_sigmoids[j]
            partial_of_loss_wrt_param /=  float(self.batch_size)

             

            moment[i] = (b1 * moment[i]) + ((1-b1) * partial_of_loss_wrt_param)
            velocity[i] = (b2 * velocity[i]) + ((1-b2) * partial_of_loss_wrt_param**2)
            m_unbias = moment[i] / (1 - b1**(train_itr+1))
            v_unbias = velocity[i] / (1 - b2**(train_itr+1))
            step = self.learning_rate * (m_unbias / math.sqrt(v_unbias + eps)) * -1 
            # step = -1 * self.learning_rate * (moment[i] / math.sqrt(velocity[i] + eps))

             
            self.vals_for_learnable_params[param] += step

        y_error_avg = sum(y_errors_in_batch) / float(self.batch_size)
        deriv_sigmoid_avg = sum(deriv_sigmoids) / float(self.batch_size)

         

        total_partial = y_error_avg * deriv_sigmoid_avg
        moment[-1] = (b1 * moment[-1]) + ((1-b1) * total_partial)
        velocity[-1] = (b2 * velocity[-1]) + ((1-b2) * total_partial**2)
        m_unbias = moment[-1] / (1 - b1**(train_itr+1))
        v_unbias = velocity[-1] / (1 - b2**(train_itr+1))
        b_step = self.learning_rate * (m_unbias / math.sqrt(v_unbias + eps))
        self.bias += b_step    

         

    def run_training_loop_multi_neuron_model(self, training_data, b1, b2, eps):

        class DataLoader:
            def __init__(self, training_data, batch_size):
                self.training_data = training_data
                self.batch_size = batch_size
                self.class_0_samples = [(item, 0) for item in self.training_data[0]]     
                self.class_1_samples = [(item, 1) for item in self.training_data[1]]     

            def __len__(self):
                return len(self.training_data[0]) + len(self.training_data[1])

            def _getitem(self):    
                cointoss = random.choice([0,1])                             
                                                                         
                if cointoss == 0:
                    return random.choice(self.class_0_samples)
                else:
                    return random.choice(self.class_1_samples)            

            def getbatch(self):
                batch_data,batch_labels = [],[]                             
                maxval = 0.0                                                
                for _ in range(self.batch_size):
                    item = self._getitem()
                    if np.max(item[0]) > maxval: 
                        maxval = np.max(item[0])
                    batch_data.append(item[0])
                    batch_labels.append(item[1])
                batch_data = [item/maxval for item in batch_data]           
                batch = [batch_data, batch_labels]
                return batch                


        self.vals_for_learnable_params = {param: random.uniform(0,1) for param in self.learnable_params}
        self.bias =   {i : [random.uniform(0,1) for j in range( self.layers_config[i] ) ]  for i in range(1, self.num_layers)}

         

        velocity_of_params = {param: 0 for param in self.learnable_params}       
        moment_of_params = {param: 0 for param in self.learnable_params}       
        bias_v = {i : [random.uniform(0,1) for j in range( self.layers_config[i] ) ]  for i in range(1, self.num_layers)}
        bias_m = {i : [random.uniform(0,1) for j in range( self.layers_config[i] ) ]  for i in range(1, self.num_layers)}

         

        data_loader = DataLoader(training_data, batch_size=self.batch_size)
        loss_running_record = []
        i = 0
        avg_loss_over_iterations = 0.0                                           
                                                                                 
        for i in range(self.training_iterations):
            data = data_loader.getbatch()
            data_tuples = data[0]
            class_labels = data[1]
            self.forward_prop_multi_neuron_model(data_tuples)                                        
            predicted_labels_for_batch = self.forw_prop_vals_at_layers[self.num_layers-1]            
            y_preds =  [item for sublist in  predicted_labels_for_batch  for item in sublist]        
            loss = sum([(abs(class_labels[i] - y_preds[i]))**2 for i in range(len(class_labels))])   
            loss_avg = loss / float(len(class_labels))                                               
            avg_loss_over_iterations += loss_avg                                                     
            if i%(self.display_loss_how_often) == 0: 
                avg_loss_over_iterations /= self.display_loss_how_often
                loss_running_record.append(avg_loss_over_iterations)
                # print("[iter=%d]  loss = %.4f" %  (i+1, avg_loss_over_iterations))                   
                avg_loss_over_iterations = 0.0                                                       
            y_errors_in_batch = list(map(operator.sub, class_labels, y_preds))
            
         

            self.backprop_and_update_params_multi_neuron_model(y_preds, y_errors_in_batch, velocity_of_params, moment_of_params, bias_v, bias_m, b1, b2, eps, i)

        return loss_running_record
    
         
        
    def backprop_and_update_params_multi_neuron_model(self, predictions, y_errors, velocity_of_param, moment_of_params, bias_v, bias_m, b1, b2, eps, train_itr):
        
        pred_err_backproped_at_layers =   [ {i : [None for j in range( self.layers_config[i] ) ]  
                                                                  for i in range(self.num_layers)} for _ in range(self.batch_size) ]
        
        partial_of_loss_wrt_params = {param : 0.0 for param in self.all_params}
    
        bias_changes =   {i : [0.0 for j in range( self.layers_config[i] ) ]  for i in range(1, self.num_layers)}
        for b in range(self.batch_size):
            pred_err_backproped_at_layers[b][self.num_layers - 1] = [ y_errors[b] ]
            for back_layer_index in reversed(range(1,self.num_layers)):              
                input_vals = self.forw_prop_vals_at_layers[back_layer_index -1]      
                deriv_sigmoids =  self.gradient_vals_for_layers[back_layer_index]    
                vars_in_layer  =  self.layer_vars[back_layer_index]                  
                vars_in_next_layer_back  =  self.layer_vars[back_layer_index - 1]    
                vals_for_input_vars_dict =  dict(zip(vars_in_next_layer_back, self.forw_prop_vals_at_layers[back_layer_index - 1][b]))   
                
                layer_params = self.layer_params[back_layer_index]         
                transposed_layer_params = list(zip(*layer_params))                   
                for k,var1 in enumerate(vars_in_next_layer_back):
                    for j,var2 in enumerate(vars_in_layer):
                        pred_err_backproped_at_layers[b][back_layer_index - 1][k] = sum([self.vals_for_learnable_params[transposed_layer_params[k][i]]
                                                                                       * pred_err_backproped_at_layers[b][back_layer_index][i]
                                                                                                                  for i in range(len(vars_in_layer))])
                for j,var in enumerate(vars_in_layer):
                    layer_params = self.layer_params[back_layer_index][j]            
                    input_vars_to_param_map = self.var_to_var_param[var]             
                    param_to_vars_map = {param : var for var, param in input_vars_to_param_map.items()}    

                    for i,param in enumerate(layer_params):
                        partial_of_loss_wrt_params[param]   +=   pred_err_backproped_at_layers[b][back_layer_index][j] * \
                                                                        vals_for_input_vars_dict[param_to_vars_map[param]] * deriv_sigmoids[b][j]
                
                for k,var1 in enumerate(vars_in_next_layer_back):
                    for j,var2 in enumerate(vars_in_layer):
                        if back_layer_index-1 > 0:
                            bias_changes[back_layer_index-1][k] += pred_err_backproped_at_layers[b][back_layer_index - 1][k] * deriv_sigmoids[b][j] 
 
         
        for param in partial_of_loss_wrt_params: 
            partial_of_loss_wrt_param = partial_of_loss_wrt_params[param] /  float(self.batch_size)   

             

            moment_of_params[param] = (b1 * moment_of_params[param]) + ((1-b1) * partial_of_loss_wrt_param)
            velocity_of_param[param] = (b2 * velocity_of_param[param]) + ((1-b2) * partial_of_loss_wrt_param**2)
            m_unbias = moment_of_params[param] / (1 - b1**(train_itr+1))
            v_unbias = velocity_of_param[param] / (1 - b2**(train_itr+1))
            step = self.learning_rate * (m_unbias / math.sqrt(v_unbias + eps))

             

            self.vals_for_learnable_params[param] += step

         
        for layer_index in range(1,self.num_layers):           
            for k in range(self.layers_config[layer_index]):
                 

                total_partial = ( bias_changes[layer_index][k] / float(self.batch_size) )
                bias_m[layer_index][k] = (b1 * bias_m[layer_index][k]) + ((1-b1) * total_partial)
                bias_v[layer_index][k] = (b2 * bias_v[layer_index][k]) + ((1-b2) * total_partial**2)
                m_unbias = bias_m[layer_index][k] / (1 - b1**(train_itr+1))
                v_unbias = bias_v[layer_index][k] / (1 - b2**(train_itr+1))
                b_step = self.learning_rate * (m_unbias / math.sqrt(v_unbias + eps)) * -1
                self.bias[layer_index][k] += b_step

                 

class EC(ComputationalGraphPrimer):
    def run_training_loop_one_neuron_model(self, training_data):

        self.vals_for_learnable_params = {param: random.uniform(0,1) for param in self.learnable_params}

        self.bias = random.uniform(0,1)                    
                                                           

        class DataLoader:

            def __init__(self, training_data, batch_size):
                self.training_data = training_data
                self.batch_size = batch_size
                self.class_0_samples = [(item, 0) for item in self.training_data[0]]    
                self.class_1_samples = [(item, 1) for item in self.training_data[1]]    

            def __len__(self):
                return len(self.training_data[0]) + len(self.training_data[1])

            def _getitem(self):    
                cointoss = random.choice([0,1])                             
                                                                            
                if cointoss == 0:
                    return random.choice(self.class_0_samples)
                else:
                    return random.choice(self.class_1_samples)            

            def getbatch(self):
                batch_data,batch_labels = [],[]                             
                maxval = 0.0                                                
                for _ in range(self.batch_size):
                    item = self._getitem()
                    if np.max(item[0]) > maxval: 
                        maxval = np.max(item[0])
                    batch_data.append(item[0])
                    batch_labels.append(item[1])
                batch_data = [item/maxval for item in batch_data]           
                batch = [batch_data, batch_labels]
                return batch                


        data_loader = DataLoader(training_data, batch_size=self.batch_size)
        loss_running_record = []
        i = 0
        avg_loss_over_iterations = 0.0                                     
                                                                            
        for i in range(self.training_iterations):
            data = data_loader.getbatch()
            data_tuples_in_batch = data[0]
            class_labels_in_batch = data[1]
            y_preds, deriv_sigmoids =  self.forward_prop_one_neuron_model(data_tuples_in_batch)      
            loss = sum([(abs(class_labels_in_batch[i] - y_preds[i]))**2 for i in range(len(class_labels_in_batch))])   
            avg_loss_over_iterations += loss / float(len(class_labels_in_batch))
            if i%(self.display_loss_how_often) == 0: 
                avg_loss_over_iterations /= self.display_loss_how_often
                loss_running_record.append(avg_loss_over_iterations)
                print("[iter=%d]  loss = %.4f" %  (i+1, avg_loss_over_iterations))                  
                avg_loss_over_iterations = 0.0                                                      
            y_errors_in_batch = list(map(operator.sub, class_labels_in_batch, y_preds))
            self.backprop_and_update_params_one_neuron_model(data_tuples_in_batch, y_preds, y_errors_in_batch, deriv_sigmoids)   
        return loss_running_record
    
    def run_training_loop_one_neuron_model_no_normal(self, training_data):

        self.vals_for_learnable_params = {param: random.uniform(0,1) for param in self.learnable_params}

        self.bias = random.uniform(0,1)                    
                                                           

        class DataLoader:

            def __init__(self, training_data, batch_size):
                self.training_data = training_data
                self.batch_size = batch_size
                self.class_0_samples = [(item, 0) for item in self.training_data[0]]    
                self.class_1_samples = [(item, 1) for item in self.training_data[1]]    

            def __len__(self):
                return len(self.training_data[0]) + len(self.training_data[1])

            def _getitem(self):    
                cointoss = random.choice([0,1])                             
                                                                            
                if cointoss == 0:
                    return random.choice(self.class_0_samples)
                else:
                    return random.choice(self.class_1_samples)            

            def getbatch(self):
                batch_data,batch_labels = [],[]                             
                # maxval = 0.0                                                
                for _ in range(self.batch_size):
                    item = self._getitem()
                    # if np.max(item[0]) > maxval: 
                    #     maxval = np.max(item[0])
                    batch_data.append(item[0])
                    batch_labels.append(item[1])
                # batch_data = [item/maxval for item in batch_data]           
                batch = [batch_data, batch_labels]
                return batch                


        data_loader = DataLoader(training_data, batch_size=self.batch_size)
        loss_running_record = []
        i = 0
        avg_loss_over_iterations = 0.0                                     
                                                                            
        for i in range(self.training_iterations):
            data = data_loader.getbatch()
            data_tuples_in_batch = data[0]
            class_labels_in_batch = data[1]
            y_preds, deriv_sigmoids =  self.forward_prop_one_neuron_model(data_tuples_in_batch)      
            loss = sum([(abs(class_labels_in_batch[i] - y_preds[i]))**2 for i in range(len(class_labels_in_batch))])   
            avg_loss_over_iterations += loss / float(len(class_labels_in_batch))
            if i%(self.display_loss_how_often) == 0: 
                avg_loss_over_iterations /= self.display_loss_how_often
                loss_running_record.append(avg_loss_over_iterations)
                print("[iter=%d]  loss = %.4f" %  (i+1, avg_loss_over_iterations))                  
                avg_loss_over_iterations = 0.0                                                      
            y_errors_in_batch = list(map(operator.sub, class_labels_in_batch, y_preds))
            self.backprop_and_update_params_one_neuron_model(data_tuples_in_batch, y_preds, y_errors_in_batch, deriv_sigmoids)   
        return loss_running_record
    
    def run_training_loop_one_neuron_model_clip(self, training_data):

        self.vals_for_learnable_params = {param: random.uniform(0,1) for param in self.learnable_params}

        self.bias = random.uniform(0,1)                    
                                                           

        class DataLoader:

            def __init__(self, training_data, batch_size):
                self.training_data = training_data
                self.batch_size = batch_size
                self.class_0_samples = [(item, 0) for item in self.training_data[0]]     
                self.class_1_samples = [(item, 1) for item in self.training_data[1]]     

            def __len__(self):
                return len(self.training_data[0]) + len(self.training_data[1])

            def _getitem(self):    
                cointoss = random.choice([0,1])                             
                                                                            
                if cointoss == 0:
                    return random.choice(self.class_0_samples)
                else:
                    return random.choice(self.class_1_samples)                        

            def clipper(self, item):
                if item[1] == 0:
                    return np.clip(item[0], -3, 7)
                else:
                    return np.clip(item[0], -6, 14)
            
            def getbatch(self):
                batch_data,batch_labels = [],[]                             
                maxval = 0.0                                                
                for i in range(self.batch_size):
                    item = self._getitem()
                    item_val = self.clipper(item)
                    if np.max(item_val) > maxval:
                        maxval = np.max(item_val)
                    batch_data.append(item_val)
                    batch_labels.append(item[1])
                batch_data = [((item/maxval * 2) - 1) for item in batch_data]           
                batch = [batch_data, batch_labels]
                return batch   


        data_loader = DataLoader(training_data, batch_size=self.batch_size)
        loss_running_record = []
        i = 0
        avg_loss_over_iterations = 0.0                                     
                                                                            
        for i in range(self.training_iterations):
            data = data_loader.getbatch()
            data_tuples_in_batch = data[0]
            class_labels_in_batch = data[1]
            y_preds, deriv_sigmoids =  self.forward_prop_one_neuron_model(data_tuples_in_batch)      
            loss = sum([(abs(class_labels_in_batch[i] - y_preds[i]))**2 for i in range(len(class_labels_in_batch))])   
            avg_loss_over_iterations += loss / float(len(class_labels_in_batch))
            if i%(self.display_loss_how_often) == 0: 
                avg_loss_over_iterations /= self.display_loss_how_often
                loss_running_record.append(avg_loss_over_iterations)
                print("[iter=%d]  loss = %.4f" %  (i+1, avg_loss_over_iterations))                  
                avg_loss_over_iterations = 0.0                                                      
            y_errors_in_batch = list(map(operator.sub, class_labels_in_batch, y_preds))
            self.backprop_and_update_params_one_neuron_model(data_tuples_in_batch, y_preds, y_errors_in_batch, deriv_sigmoids)   
        return loss_running_record
    
    def run_training_loop_multi_neuron_model(self, training_data):

        class DataLoader:

            def __init__(self, training_data, batch_size):
                self.training_data = training_data
                self.batch_size = batch_size
                self.class_0_samples = [(item, 0) for item in self.training_data[0]]     
                self.class_1_samples = [(item, 1) for item in self.training_data[1]]     

            def __len__(self):
                return len(self.training_data[0]) + len(self.training_data[1])

            def _getitem(self):    
                cointoss = random.choice([0,1])                             
                                                                            
                if cointoss == 0:
                    return random.choice(self.class_0_samples)
                else:
                    return random.choice(self.class_1_samples)            

            def getbatch(self):
                batch_data,batch_labels = [],[]                             
                maxval = 0.0                                                
                for _ in range(self.batch_size):
                    item = self._getitem()
                    if np.max(item[0]) > maxval: 
                        maxval = np.max(item[0])
                    batch_data.append(item[0])
                    batch_labels.append(item[1])
                batch_data = [item/maxval for item in batch_data]           
                batch = [batch_data, batch_labels]
                return batch                

         
         
         
         
        self.vals_for_learnable_params = {param: random.uniform(0,1) for param in self.learnable_params}
         
         
        self.bias =   {i : [random.uniform(0,1) for j in range( self.layers_config[i] ) ]  for i in range(1, self.num_layers)}
        data_loader = DataLoader(training_data, batch_size=self.batch_size)
        loss_running_record = []
        i = 0
        avg_loss_over_iterations = 0.0                                           
                                                                                 
        for i in range(self.training_iterations):
            data = data_loader.getbatch()
            data_tuples = data[0]
            class_labels = data[1]
            self.forward_prop_multi_neuron_model(data_tuples)                                        
            predicted_labels_for_batch = self.forw_prop_vals_at_layers[self.num_layers-1]            
            y_preds =  [item for sublist in  predicted_labels_for_batch  for item in sublist]        
            loss = sum([(abs(class_labels[i] - y_preds[i]))**2 for i in range(len(class_labels))])   
            loss_avg = loss / float(len(class_labels))                                               
            avg_loss_over_iterations += loss_avg                                                     
            if i%(self.display_loss_how_often) == 0: 
                avg_loss_over_iterations /= self.display_loss_how_often
                loss_running_record.append(avg_loss_over_iterations)
                print("[iter=%d]  loss = %.4f" %  (i+1, avg_loss_over_iterations))                   
                avg_loss_over_iterations = 0.0                                                       
            y_errors_in_batch = list(map(operator.sub, class_labels, y_preds))
            self.backprop_and_update_params_multi_neuron_model(y_preds, y_errors_in_batch) 
        return loss_running_record
    
    def run_training_loop_multi_neuron_model_no_normal(self, training_data):

        class DataLoader:

            def __init__(self, training_data, batch_size):
                self.training_data = training_data
                self.batch_size = batch_size
                self.class_0_samples = [(item, 0) for item in self.training_data[0]]     
                self.class_1_samples = [(item, 1) for item in self.training_data[1]]     

            def __len__(self):
                return len(self.training_data[0]) + len(self.training_data[1])

            def _getitem(self):    
                cointoss = random.choice([0,1])                             
                                                                            
                if cointoss == 0:
                    return random.choice(self.class_0_samples)
                else:
                    return random.choice(self.class_1_samples)            

            def getbatch(self):
                batch_data,batch_labels = [],[]                             
                # maxval = 0.0                                                
                for _ in range(self.batch_size):
                    item = self._getitem()
                    # if np.max(item[0]) > maxval: 
                    #     maxval = np.max(item[0])
                    batch_data.append(item[0])
                    batch_labels.append(item[1])
                # batch_data = [item/maxval for item in batch_data]           
                batch = [batch_data, batch_labels]
                return batch                

         
         
         
         
        self.vals_for_learnable_params = {param: random.uniform(0,1) for param in self.learnable_params}
         
         
        self.bias =   {i : [random.uniform(0,1) for j in range( self.layers_config[i] ) ]  for i in range(1, self.num_layers)}
        data_loader = DataLoader(training_data, batch_size=self.batch_size)
        loss_running_record = []
        i = 0
        avg_loss_over_iterations = 0.0                                           
                                                                                 
        for i in range(self.training_iterations):
            data = data_loader.getbatch()
            data_tuples = data[0]
            class_labels = data[1]
            self.forward_prop_multi_neuron_model(data_tuples)                                        
            predicted_labels_for_batch = self.forw_prop_vals_at_layers[self.num_layers-1]            
            y_preds =  [item for sublist in  predicted_labels_for_batch  for item in sublist]        
            loss = sum([(abs(class_labels[i] - y_preds[i]))**2 for i in range(len(class_labels))])   
            loss_avg = loss / float(len(class_labels))                                               
            avg_loss_over_iterations += loss_avg                                                     
            if i%(self.display_loss_how_often) == 0: 
                avg_loss_over_iterations /= self.display_loss_how_often
                loss_running_record.append(avg_loss_over_iterations)
                print("[iter=%d]  loss = %.4f" %  (i+1, avg_loss_over_iterations))                   
                avg_loss_over_iterations = 0.0                                                       
            y_errors_in_batch = list(map(operator.sub, class_labels, y_preds))
            self.backprop_and_update_params_multi_neuron_model(y_preds, y_errors_in_batch)  
        return loss_running_record

    def run_training_loop_multi_neuron_model_clip(self, training_data):

        class DataLoader:

            def __init__(self, training_data, batch_size):
                self.training_data = training_data
                self.batch_size = batch_size
                self.class_0_samples = [(item, 0) for item in self.training_data[0]]     
                self.class_1_samples = [(item, 1) for item in self.training_data[1]]     

            def __len__(self):
                return len(self.training_data[0]) + len(self.training_data[1])

            def _getitem(self):    
                cointoss = random.choice([0,1])                             
                                                                            
                if cointoss == 0:
                    return random.choice(self.class_0_samples)
                else:
                    return random.choice(self.class_1_samples)               
            
            def clipper(self, item):
                if item[1] == 0:
                    return np.clip(item[0], -3, 7)
                else:
                    return np.clip(item[0], -6, 14)
                return False
            
            def getbatch(self):
                batch_data,batch_labels = [],[]                             
                maxval = 0.0                                                
                for i in range(self.batch_size):
                    item = self._getitem()
                    item_val = self.clipper(item)
                    if np.max(item_val) > maxval:
                        maxval = np.max(item_val)
                    batch_data.append(item_val)
                    batch_labels.append(item[1])
                batch_data = [((item/maxval * 2) - 1) for item in batch_data]           
                batch = [batch_data, batch_labels]
                return batch                

         
        self.vals_for_learnable_params = {param: random.uniform(0,1) for param in self.learnable_params}
         
         
        self.bias =   {i : [random.uniform(0,1) for j in range( self.layers_config[i] ) ]  for i in range(1, self.num_layers)}
        data_loader = DataLoader(training_data, batch_size=self.batch_size)
        loss_running_record = []
        i = 0
        avg_loss_over_iterations = 0.0                                           
                                                                                 
        for i in range(self.training_iterations):
            data = data_loader.getbatch()
            data_tuples = data[0]
            class_labels = data[1]
            self.forward_prop_multi_neuron_model(data_tuples)                                        
            predicted_labels_for_batch = self.forw_prop_vals_at_layers[self.num_layers-1]            
            y_preds =  [item for sublist in  predicted_labels_for_batch  for item in sublist]        
            loss = sum([(abs(class_labels[i] - y_preds[i]))**2 for i in range(len(class_labels))])   
            loss_avg = loss / float(len(class_labels))                                               
            avg_loss_over_iterations += loss_avg                                                     
            if i%(self.display_loss_how_often) == 0: 
                avg_loss_over_iterations /= self.display_loss_how_often
                loss_running_record.append(avg_loss_over_iterations)
                print("[iter=%d]  loss = %.4f" %  (i+1, avg_loss_over_iterations))                   
                avg_loss_over_iterations = 0.0                                                       
            y_errors_in_batch = list(map(operator.sub, class_labels, y_preds))
            self.backprop_and_update_params_multi_neuron_model(y_preds, y_errors_in_batch) 
        return loss_running_record