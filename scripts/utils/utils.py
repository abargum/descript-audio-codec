import torch

def load_dict_from_txt(file_path):
    data = {}
    with open(file_path, 'r') as file:
        lines = file.readlines()
        
        # Extract headers (e.g., ID, AGE, etc.)
        headers = lines[0].strip().split()  
        
        # Process each subsequent line
        for line in lines[1:]:
            values = line.strip().split(maxsplit=len(headers) - 1)
            record_id = "p" + values[0]  
            record = dict(zip(headers[1:], values[1:])) 
            record["AGE"] = int(record["AGE"]) 
            data[record_id] = record  
    
    return data

def load_speaker_statedict(path):
        loaded_state = torch.load(path, map_location="cuda:%d" % 0)
        
        newdict = {}
        pqmfdict = {}
        delete_list = []
        
        for name, param in loaded_state.items():
            new_name = name.replace("__S__.", "")
            
            if "pqmf" in new_name:
                new_name = new_name.replace("pqmf.", "")
                pqmfdict[new_name] = param
            else:
                newdict[new_name] = param
                
            delete_list.append(name)
        loaded_state.update(newdict)
        for name in delete_list:
            del loaded_state[name]
                
        return loaded_state, pqmfdict