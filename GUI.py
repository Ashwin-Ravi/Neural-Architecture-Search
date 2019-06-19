import tkinter
from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from PIL import Image, ImageTk
from tkinter.filedialog import askopenfilename
import os



def trainSearch(data_path):
    global batch_size #160
    global epoch #150
    global num_layer #6
    global num_cells #5
    os.system("export PYTHONPATH=./NAO/NAO-WS/cnn:$PYTHONPATH")
    os.system("MODEL=search_cifar10")
    os.system("MODEL_DIR=NAO/NAO-WS/cnn/models/$MODEL")
    os.system("LOG_DIR=NAO/NAO-WS/cnn/logs")
    os.system("DATA_DIR=NAO/NAO-WS/cnn/data/cifar10")
    model = "search_cifar10"
    MODEL_DIR = "NAO/NAO-WS/cnn/models/$MODEL"
    LOG_DIR = "NAO/NAO-WS/cnn/logs"
    DATA_DIR= data_path
    run_str = "python ./NAO/NAO-WS/cnn/train_search.py --child_data_format='NCHW' --data_path=" + DATA_DIR+" --output_dir=" + MODEL_DIR +" --child_sample_policy=uniform --child_batch_size="+str(batch_size.get())+" --child_num_epochs="+str(epoch.get())+" --child_eval_every_epochs=30 --child_use_aux_heads --child_num_layers="+str(num_layer.get())+" --child_out_filters=20 --child_l2_reg=1e-4 --child_num_cells="+str(num_cells.get())+" --child_keep_prob=0.90 --child_drop_path_keep_prob=0.60 --child_lr_cosine --child_lr_max=0.05 --child_lr_min=0.0005 --child_lr_T_0=10 --child_lr_T_mul=2 --child_eval_batch_size=500 --controller_encoder_vocab_size=12 --controller_decoder_vocab_size=12 --controller_encoder_emb_size=48 --controller_encoder_hidden_size=96 --controller_decoder_hidden_size=96 --controller_mlp_num_layers=3 --controller_mlp_hidden_size=100 --controller_mlp_dropout=0.1 --controller_source_length=40 --controller_encoder_length=20 --controller_decoder_length=40 --controller_train_epochs=1000 --controller_optimizer=adam --controller_lr=0.001 --controller_batch_size=100 --controller_save_frequency=100 --controller_attention --controller_time_major --controller_symmetry 2>&1 | tee -a NAO/NAO-WS/cnn/logs/train.search_cifar10.log"

    os.system(run_str)
    
    
window = Tk()
 
window.title("Neural Architecture Search")
window.geometry('1000x500')


tab_control = ttk.Notebook(window)
tab1 = ttk.Frame(tab_control)
tab2 = ttk.Frame(tab_control)
tab3 = ttk.Frame(tab_control)
tab_control.add(tab1, text='Search Architecture')
tab_control.add(tab2, text='Train Discovered Architectures')
tab_control.add(tab3, text='Test Architecture')

lbl = Label(tab1, text="Select Dataset")
lbl.grid(column=0, row=0)

v = tkinter.IntVar()
v.set(10)

data_path_arr = ["NAO/NAO-WS/cnn/data/cifar10", "NAO/NAO-WS/cnn/data/cifar100", ""]
folder_path = StringVar()
def browse_button():
    # Allow user to select a directory and store it in global var
    # called folder_path
    global folder_path
    global data_path_arr
    filename = filedialog.askdirectory()
    folder_path.set(filename)
    data_path_arr[2] = str(filename)

rad1 = Radiobutton(tab1,text='CIFAR-10', variable=v, value=0)
rad2 = Radiobutton(tab1,text='CIFAR-100', variable=v, value=1)
rad3 = Radiobutton(tab1,text='Custom Dataset', variable=v, value=2)
browse_button = Button(tab1, text ='Browse', command = browse_button)

rad1.grid(column=0, row=1, sticky="W")
rad2.grid(column=0, row=2, sticky="W")
rad3.grid(column=0, row=3, sticky="W")
browse_button.grid(column=1, row=3)


batch_size = StringVar()
epoch = StringVar()
num_layer = StringVar()
num_cells = StringVar()



lbl1 = Label(tab1, text="Select Hyperparameters")
lbl1.grid(column=3, row=0)

lbl2 = Label(tab1, text="Batch Size")
lbl2.grid(column=3, row=1, sticky="E")

txt1 = Entry(tab1,width=4, textvariable = batch_size)
txt1.grid(column=4, row=1, sticky="W")


lbl3 = Label(tab1, text="Epochs")
lbl3.grid(column=3, row=2, sticky="E")

txt2 = Entry(tab1,width=4, textvariable = epoch)
txt2.grid(column=4, row=2, sticky="W")


lbl3b = Label(tab1, text="Num of Layers")
lbl3b.grid(column=3, row=3, sticky="E")

txt2b = Entry(tab1,width=4, textvariable = num_layer)
txt2b.grid(column=4, row=3, sticky="W")

lbl3a = Label(tab1, text="Num of Cells")
lbl3a.grid(column=3, row=4, sticky="E")

txt2a = Entry(tab1,width=4, textvariable = num_cells)
txt2a.grid(column=4, row=4, sticky="W")

def loadData():
    data_arr = ['./CIFAR.png', './CIFAR100.png', './CIFAR100.png']
    global data_path_arr
    rad_num = v.get()
    data_path = data_path_arr[rad_num]
    load = Image.open(data_arr[rad_num])
    w, h = load.size
    load = load.resize((5*w, 5*h))
    imgfile = ImageTk.PhotoImage(load )
    
    canvas.image = imgfile  # <--- keep reference of your image
    canvas.create_image(2,2,anchor='nw',image=imgfile)
    

#load_button = Button(tab1, text ='Load Data')
load_button = Button(tab1, text ='Load Data', command = loadData)
load_button.grid(column=0, row=5, sticky="W")

lbl4 = Label(tab1, text="    ")
lbl4.grid(column=0, row=6, sticky="W")


lbl5 = Label(tab1, text="Sample Input Dataset")
lbl5.grid(column=0, row=7, sticky="W")


canvas = Canvas(tab1, width=160,height=160, bd=0,bg='white')
canvas.grid(row=8, column=0)

def searchArch():
    global batch_size
    global epoch
    global num_layer
    global num_cells
    textvar = "Current status:"
    t_status.delete(0.0, tkinter.END)
    t_status.insert('insert', textvar+'Searching for optimal architectures')
    t_status.update()
    global data_path_arr
    rad_num = v.get()
    data_path = data_path_arr[rad_num]
    trainSearch(data_path)
    t_status.insert('insert', textvar+'Done. Check output / error log')
    

search_button = Button(tab1, text ='Search Architecture', command = searchArch)
# search_button = Button(window, text ='Load Data', command = loadData)
search_button.grid(column=5, row=5, pady=50)

lbl6 = Label(tab1, text="Search status")
lbl6.grid(column=6, row=7, sticky="W")


#canvasout = Canvas(tab1, width=160,height=160, bd=0,bg='white')
#canvasout.grid(row=8, column=5)

t_status=Text(tab1,bd=0, width=40,height=10,font='Fixdsys -14')
t_status.grid(row=8, column=6)

#save_button = Button(tab1, text ='Save Model')
# search_button = Button(window, text ='Load Data', command = loadData)
#save_button.grid(column=6, row=9)




# tab 2



def trainFinal():
    global final_batch_size #144
    global final_epoch #630
    global final_num_layer #15
    global final_num_cells #5
    global final_num_branches #5
    os.system("export PYTHONPATH=./NAO/NAO-WS/cnn:$PYTHONPATH")
    model = "search_cifar10"
    MODEL_DIR = "NAO/NAO-WS/cnn/models/$MODEL"
    LOG_DIR = "NAO/NAO-WS/cnn/logs"
    DATA_DIR= "NAO/NAO-WS/cnn/data/cifar10"
    fixed_arc="0 1 0 4 2 2 1 3 1 0 2 0 2 3 1 3 1 2 0 4 1 1 0 3 2 3 2 3 1 0 0 1 0 4 2 4 2 3 2 3"
    run_str = "python NAO/NAO-WS/cnn/train.py --data_path="+DATA_DIR+" --output_dir="+MODEL_DIR+" --child_data_format='NCHW' --child_batch_size="+str(final_batch_size.get())+" --child_num_epochs="+str(final_epoch.get())+" --child_eval_every_epochs=1 --child_fixed_arc="+fixed_arc+" --child_use_aux_heads --child_num_layers="+str(final_num_layer.get())+" --child_out_filters=36 --child_num_branches="+str(final_num_branches.get())+" --child_num_cells="+str(final_num_cells.get())+" --child_keep_prob=0.8 --child_drop_path_keep_prob=0.6 --child_l2_reg=2e-4 --child_lr_cosine --child_lr_max=0.05 --child_lr_min=0.0001 --child_lr_T_0=10 --child_lr_T_mul=2 2>&1 | tee -a "+LOG_DIR+"/train.log"
    
    os.system(run_str)


def finalSearch():
    textvar = "Current status:"
    t2_status.delete(0.0, tkinter.END)
    t2_status.insert('insert', textvar+'Training Discovered Architectures')
    t2_status.update()
    trainFinal()
    t2_status.insert('insert', textvar+'Done. Check output / error log')

final_batch_size = StringVar()
final_epoch = StringVar()
final_num_layer = StringVar()
final_num_cells = StringVar()
final_num_branches = StringVar()

row_num = 0


Label(tab2, text="Select Hyperparameters").grid(column=1, row=0)
Label(tab2, text="Batch Size").grid(column=1, row=1, sticky="E")
Entry(tab2,width=4, textvariable = final_batch_size).grid(column=2, row=1, sticky="W")
Label(tab2, text="Epochs").grid(column=1, row=2, sticky="E")
Entry(tab2,width=4, textvariable = final_epoch).grid(column=2, row=2, sticky="W")
Label(tab2, text="Num of Layers").grid(column=1, row=3, sticky="E")
Entry(tab2,width=4, textvariable = final_num_layer).grid(column=2, row=3, sticky="W")
Label(tab2, text="Num of Cells").grid(column=1, row=4, sticky="E")
Entry(tab2,width=4, textvariable = final_num_cells).grid(column=2, row=4, sticky="W")
Label(tab2, text="Num of branches").grid(column=1, row=5, sticky="E")
Entry(tab2,width=4, textvariable = final_num_branches).grid(column=2, row=5, sticky="W")


Button(tab2, text ='Train Architectures', command = finalSearch).grid(column=1, row=6, pady=50)
# search_button = Button(window, text ='Load Data', command = loadData)


Label(tab2, text="Train Discovered Architectures").grid(column=1, row=7, sticky="W")


#canvasout = Canvas(tab1, width=160,height=160, bd=0,bg='white')
#canvasout.grid(row=8, column=5)

t2_status=Text(tab2,bd=0, width=40,height=10,font='Fixdsys -14')
t2_status.grid(row=8, column=1)






# tab 3



def test():
    global test_batch_size #144
    global test_epoch #630
    global test_num_layer #15
    global test_num_cells #5
    global test_num_branches #5
    os.system("export PYTHONPATH=./NAO/NAO-WS/cnn:$PYTHONPATH")
    model = "search_cifar10"
    MODEL_DIR = "NAO/NAO-WS/cnn/models/$MODEL"
    LOG_DIR = "NAO/NAO-WS/cnn/logs"
    DATA_DIR= "NAO/NAO-WS/cnn/data/cifar10"
    fixed_arc="0 1 0 4 2 2 1 3 1 0 2 0 2 3 1 3 1 2 0 4 1 1 0 3 2 3 2 3 1 0 0 1 0 4 2 4 2 3 2 3"
    run_str = "python NAO/NAO-WS/cnn/test.py --data_path="+DATA_DIR+" --output_dir="+MODEL_DIR+" --child_data_format='NCHW' --child_batch_size="+str(test_batch_size.get())+" --child_num_epochs="+str(test_epoch.get())+" --child_eval_every_epochs=1 --child_fixed_arc="+fixed_arc+" --child_use_aux_heads --child_num_layers="+str(test_num_layer.get())+" --child_out_filters=36 --child_num_branches="+str(test_num_branches.get())+" --child_num_cells="+str(test_num_cells.get())+" --child_keep_prob=0.8 --child_drop_path_keep_prob=0.6 --child_l2_reg=2e-4 --child_lr_cosine --child_lr_max=0.05 --child_lr_min=0.0001 --child_lr_T_0=10 --child_lr_T_mul=2"
    
    os.system(run_str)


def finalTest():
    textvar = "Current status:"
    t3_status.delete(0.0, tkinter.END)
    t3_status.insert('insert', textvar+'Testing Architecture')
    t3_status.update()
    test()
    t3_status.insert('insert', textvar+'Done. Check output / error log')

test_batch_size = StringVar()
test_epoch = StringVar()
test_num_layer = StringVar()
test_num_cells = StringVar()
test_num_branches = StringVar()

row_num = 0


Label(tab3, text="Select Hyperparameters").grid(column=1, row=0)
Label(tab3, text="Batch Size").grid(column=1, row=1, sticky="E")
Entry(tab3,width=4, textvariable = test_batch_size).grid(column=2, row=1, sticky="W")
Label(tab3, text="Epochs").grid(column=1, row=2, sticky="E")
Entry(tab3,width=4, textvariable = test_epoch).grid(column=2, row=2, sticky="W")
Label(tab3, text="Num of Layers").grid(column=1, row=3, sticky="E")
Entry(tab3,width=4, textvariable = test_num_layer).grid(column=2, row=3, sticky="W")
Label(tab3, text="Num of Cells").grid(column=1, row=4, sticky="E")
Entry(tab3,width=4, textvariable = test_num_cells).grid(column=2, row=4, sticky="W")
Label(tab3, text="Num of branches").grid(column=1, row=5, sticky="E")
Entry(tab3,width=4, textvariable = test_num_branches).grid(column=2, row=5, sticky="W")


Button(tab3, text ='Test', command = finalTest).grid(column=1, row=6, pady=50)
# search_button = Button(window, text ='Load Data', command = loadData)


Label(tab3, text="Test Architecture").grid(column=1, row=7, sticky="W")


#canvasout = Canvas(tab1, width=160,height=160, bd=0,bg='white')
#canvasout.grid(row=8, column=5)

t3_status=Text(tab3,bd=0, width=40,height=10,font='Fixdsys -14')
t3_status.grid(row=8, column=1)


#lblt1 = Label(tab2, text="Load Model")
#lblt1.grid(column=0, row=0, sticky="W")

#load_model_button = Button(tab2, text ='Browse')
## load_button = Button(window, text ='Load Data', command = loadModel)
#load_model_button.grid(column=1, row=0, sticky="W")

#lblt2 = Label(tab2, text="image")
#lblt2.grid(column=1, row=2)

#canvas3 = Canvas(tab2, width=160,height=160, bd=0,bg='white')
#canvas3.grid(row=3, column=1)

#lblt3 = Label(tab2, text="Load image")
#lblt3.grid(column=0, row=1, sticky="W")

#e = StringVar()

#def showImg():
#    File = askopenfilename(title='Open Image') 
#    e.set(File)
#    
#    load = Image.open(e.get())
#    w, h = load.size
#    load = load.resize((5*w, 5*h))
#    imgfile = ImageTk.PhotoImage(load )
    
#    canvas3.image = imgfile  # <--- keep reference of your image
#    canvas3.create_image(2,2,anchor='nw',image=imgfile)

#load_image_button = Button(tab2, text ='Browse', command = showImg)
## load_button = Button(window, text ='Load Data', command = loadModel)
#load_image_button.grid(column=1, row=1, sticky="W")


#def Predict():
#    textvar = "The object is :"
#    t1.delete(0.0, tkinter.END)
#    t1.insert('insert', textvar+'\n')
#    t1.update()

#submit_button = Button(tab2, text ='Predict')
#submit_button = Button(tab2, text ='Predict', command = Predict)
#submit_button.grid(row=2, column=2)

#t1=Text(tab2,bd=0, width=20,height=10,font='Fixdsys -14')
#t1.grid(row=3, column=2)

tab_control.pack(expand=1, fill='both')
window.mainloop()
