# FuzzSIM

## Project structure
```txt
FuzzSIM/
│── src/            
│   ├── classif/
│   │   ├── base.py
│   │   ├── kmm.py              
│   ├── utils/     
│   │   ├── dataloader.py              
│   │   ├── set.py              
│   │   ├── utils.py              
│   │   ├── visualization.py    
│   ├── fsim.py
│── configs/               
│   ├── config.yml    
│── bin/                 
│── scripts/                
│   ├── launch.sh     
│── environment.yml
│── README.md               
│── .gitignore  
│── .gitattributes             
```


## Setup conda environment
```bash
conda env create -f environment.yml
conda activate fuzzsim
```
