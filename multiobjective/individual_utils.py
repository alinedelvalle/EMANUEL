import numpy as np

class IndividualUtils:
    
    # Decodificação do indivíduo:
    # Os hip. booleanos aparecem ou não no comando, mas ocupam posições no indivíduo
    # Os hip. dependentes de condicionais são contabilizados no indivíduo a depender
    # da condição
    

    # -------------------------------------------------------------------------
    # Decoding function for: MULAN
    # -------------------------------------------------------------------------
    def command_mulan(individual, algorithm, config_algorithm, i):
        # example of algorithm: meka.classifiers.multilabel.MULAN.MLkNN
        mlc = algorithm.split('.')[-1]
        meka_command = 'meka.classifiers.multilabel.MULAN -S ' + mlc

        is_normalize = False

        for var, all_values in config_algorithm.items():

            if var == '-normalize':
                value = all_values[individual[0]%len(all_values)]
                is_normalize = False if value%2 == 0 else True
                # does not increase i
            else:
                value = all_values[individual[i]%len(all_values)]
                meka_command += '.'+str(value)
                i += 1
        
        return i, is_normalize, meka_command
    # -------------------------------------------------------------------------


    # Função de decodificação para: ensemble de MLC, MLC e ensemble de SLC
    def command_aux(individual, config_algorithm, i):
        command = ''
        params = {}
        algorithm = ''
        is_normalize = None
        for variable in config_algorithm.keys():
            if variable == 'if': # função
                function = config_algorithm[variable]
                return_function = function(params)
        
                if isinstance(return_function, dict):
                    dictionary = return_function
                    for var in dictionary.keys():
                        all_values = dictionary[var]
                        value = all_values[individual[i]%len(all_values)]
                        params[var] = value
                        
                        if isinstance(value, np.bool_):
                            if value==True:
                                command = command + ' ' + var
                        else:
                                command = command + ' ' + var + ' ' + str(value)
                        
                        i = i + 1  
            else:
                all_values = config_algorithm[variable]
                
                if variable == '-normalize':
                    value = all_values[individual[0]%len(all_values)]
                    is_normalize = False if value%2 == 0 else True
                    # does not increase i
                elif variable == '-W':
                    all_values = list(all_values.keys())
                    
                    value = all_values[individual[i]%len(all_values)]
                    algorithm = value

                    i = i + 1
                else:
                    value = all_values[individual[i]%len(all_values)]
                    params[variable] = value
                    
                    if isinstance(value, np.bool_):
                        if value==True:
                            command = command + ' ' + variable
                    else:
                        command = command + ' ' + variable + ' ' + str(value)
                    
                    i = i + 1
            
        return i, is_normalize, command, algorithm
    
    # Função de decodificação para SLC
    def command_slc_aux(individual, config, config_algorithm, i):
        command = ''
        params = {}
        is_normalize = None
        for variable in config_algorithm.keys():
            if variable == 'if': # função
                function = config_algorithm[variable]
                return_function = function(params)
        
                if isinstance(return_function, dict):
                    dictionary = return_function
                    for var in dictionary.keys():
                        all_values = dictionary[var]
                        value = all_values[individual[i]%len(all_values)]
                        params[var] = value
                        
                        if isinstance(value, np.bool_):
                            if value==True:
                                command = command + ' ' + var
                        else:
                            command = command + ' ' + var + ' ' + str(value)
                                
                        i = i + 1    
            else: # lista ou kernel
                all_values = config_algorithm[variable]
            
                if variable == '-normalize':
                    value = all_values[individual[0]%len(all_values)]
                    is_normalize = False if value%2 == 0 else True
                elif isinstance(all_values, np.ndarray): # lista
                    value = all_values[individual[i]%len(all_values)]
                    params[variable] = value
                    
                    if isinstance(value, np.bool_):
                        if value==True:
                            command = command + ' ' + variable
                    else:
                        command = command + ' ' + variable + ' ' + str(value)
                        
                    i = i + 1
                else: # kernel
                    kernels = list(all_values.keys()) 
                    kernel = kernels[individual[i]%len(kernels)]
                    config_kernel = config.get_sl_kernel_config().get(kernel)                       

                    command = command + ' ' + variable + ' \"' + kernel
                    i = i + 1
                
                    for var in config_kernel.keys():
                        all_values = config_kernel[var]
                        value = all_values[individual[i]%len(all_values)]
                        params[var] = value
                        
                        if isinstance(value, np.bool_):
                            if value==True:
                                command = command + ' ' + var
                        else:
                            command = command + ' ' + var + ' ' + str(value)
                            
                        i = i + 1
                              
                    command = command + '\"'
                    
        return i, is_normalize, command

    
    # Decodifica o indivíduo em comandos weka e meka
    def get_commands(config, individual):
        is_pt = False # PT - problem transformation
        is_mlc_ensemble = False
        
        # index 0
        # normalização, depende do algoritmo MLC ou SLC
        
        # index 1
        index_algorithm = individual[1] 
        algorithm = config.get_ml_algorithms()[index_algorithm%(len(config.get_ml_algorithms()))]
        
        # other indexes
        meka_command = algorithm
        weka_command = None
        i = 2
        
        # MULAN algorithms 
        if 'MULAN' in algorithm:
            config_ml = config.get_ml_config() 
            config_algorithm = config_ml.get(algorithm)
            i, is_normalize, meka_command = IndividualUtils.command_mulan(individual, algorithm, config_algorithm, i)
            
        # MEKA algorithms
        else:
            # ensemble
            if 'meta' in algorithm:
                is_mlc_ensemble = True
                config_ml = config.get_ml_ensemble_config()
                config_algorithm = config_ml.get(algorithm)
                
                i, is_normalize, command, algorithm = IndividualUtils.command_aux(individual, config_algorithm, i)
                meka_command = meka_command + command + ' -W ' + algorithm + ' --'
                config_algorithm = config_algorithm['-W'][algorithm]
                
            # MLC           
            if is_mlc_ensemble == False:
                config_ml = config.get_ml_config() 
                config_algorithm = config_ml.get(algorithm)
            
            if '-W' in config_algorithm.keys():
                is_pt = True
            
            i, is_normalize, command, algorithm = IndividualUtils.command_aux(individual, config_algorithm, i)
            meka_command = meka_command + command
            
            # SLC ensemble
            if is_pt == True:
                weka_command = algorithm + ' --'
                config_algorithm = config_algorithm['-W'][algorithm]
                
                # ensemble
                if 'meta' in algorithm or 'LWL' in algorithm:
                    i, is_normalize, command, algorithm = IndividualUtils.command_aux(individual, config_algorithm, i)
                    weka_command = weka_command + command + ' -W ' + algorithm + ' --'
                    config_algorithm = config_algorithm['-W'][algorithm]
            
                # SLC
                i, is_normalize, command = IndividualUtils.command_slc_aux(individual, config, config_algorithm, i)
                weka_command = weka_command + command
           
        #return i, is_normalize, meka_command, weka_command
        return is_normalize, meka_command, weka_command
                            
    # -------------------------------------------------------------------------
    
    # Obtém a quantidade de genes que representam o indivíduo
    # utiliza as funções auxiliares: get_lenght_aux e get_lenght_slc_aux
    def get_lenght_individual(config, individual):
        is_pt = False # PT - problem transformation
        is_mlc_ensemble = False
        
        # index 0 - normalize
        
        # index 1
        index_algorithm = individual[1] 
        algorithm = config.get_ml_algorithms()[index_algorithm%(len(config.get_ml_algorithms()))]
        
        # other indexes
        i = 2
        
        # ensemble
        if 'meta' in algorithm:
            is_mlc_ensemble = True
            config_ml = config.get_ml_ensemble_config()
            config_algorithm = config_ml.get(algorithm)
            
            i, algorithm = IndividualUtils.get_lenght_aux(individual, config_algorithm, i)
            config_algorithm = config_algorithm['-W'][algorithm]
            
        # MLC           
        if is_mlc_ensemble == False:
            config_ml = config.get_ml_config()
            config_algorithm = config_ml.get(algorithm)
        
        if '-W' in config_algorithm.keys():
            is_pt = True
        
        i, algorithm = IndividualUtils.get_lenght_aux(individual, config_algorithm, i)
        
        # SLC ensemble
        if is_pt == True:
            config_algorithm = config_algorithm['-W'][algorithm]
            
            # ensemble
            if 'meta' in algorithm or 'LWL' in algorithm:
                i, algorithm = IndividualUtils.get_lenght_aux(individual, config_algorithm, i)
                config_algorithm = config_algorithm['-W'][algorithm]
        
            # SLC
            i = IndividualUtils.get_lenght_slc_aux(individual, config, config_algorithm, i)
            
        return i # tamanho
    
    
    # Obtém a quantidade de genes ocupadas por: ensemble de MLC, MLC e ensemble de SLC
    def get_lenght_aux(individual, config_algorithm, i):
        #print(i)
        params = {}
        algorithm = ''
        for variable, all_values in config_algorithm.items():
            #print(variable)

            if variable == 'if': # função
                function = config_algorithm[variable]
                return_function = function(params)
        
                if isinstance(return_function, dict):
                    dictionary = return_function
                    for var, values in dictionary.items():
                        value = values[individual[i]%len(values)]
                        params[var] = value
                        i = i + 1  
            else:
                if variable == '-normalize':
                    continue
                    # does not count in size
                elif variable == '-W': # -W (algorithm)
                    all_values = list(all_values.keys())
                    value = all_values[individual[i]%len(all_values)]
                    algorithm = value
                else: 
                    value = all_values[individual[i]%len(all_values)]
                    params[variable] = value
                    
                i = i + 1
            
        return i, algorithm
    
    
    # Obtém a quantidade de genes ocupadas por algoritmos SLC
    def get_lenght_slc_aux(individual, config, config_algorithm, i):
        params = {}
        for variable, all_values in config_algorithm.items():
            #print(variable)
            if variable == 'if': # função
                function = config_algorithm[variable]
                return_function = function(params)
        
                if isinstance(return_function, dict):
                    dictionary = return_function
                    for var, values in dictionary.keys():
                        value = values[individual[i]%len(values)]
                        params[var] = value
                        i = i + 1    
            else: # lista ou kernel
                if variable == '-normalize':
                    continue
                    # does not count in size
            
                elif isinstance(all_values, np.ndarray): # lista
                    value = all_values[individual[i]%len(all_values)]
                    params[variable] = value  
                    i = i + 1
                    
                else: # kernel
                    kernels = list(all_values.keys()) 
                    kernel = kernels[individual[i]%len(kernels)]
                    config_kernel = config.get_sl_kernel_config().get(kernel)                       
                    i = i + 1
                
                    for var, values in config_kernel.items():
                        value = values[individual[i]%len(values)]
                        params[var] = value
                        i = i + 1
                              
        return i
    
    
    