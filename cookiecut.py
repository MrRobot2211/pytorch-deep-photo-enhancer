import os
folder = "models"

subfolders = [ f.path for f in os.scandir(folder) if f.is_dir() ]
sub2folders = ['1Way','2Way']
for f in subfolders:
    for subname in sub2folders:
        os.mkdir(f+'/'+subname)


#print(subfolders)



#os.mkdir(path)