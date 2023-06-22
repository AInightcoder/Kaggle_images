import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

import torch
import torch.nn as nn

from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, MessageHandler, filters

# pillow vs cv2

from PIL import Image
import cv2
import os

# model with nn
pickle_file = 'model_pickle.pth'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.linear = nn.Linear(28*28, 10)
    def forward(self, x):
        out = self.linear(x)
        return out


model = MLP().to(device)
model.load_state_dict(torch.load(pickle_file))
model.eval()

class TestDataset(Dataset):
    
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, ind):
        x = self.data[ind] / 255.0
        return x
    
def img2int(path):
        image = cv2.imread(path ,cv2.IMREAD_GRAYSCALE)
        image = cv2.bitwise_not(image)
        rimg = cv2.resize(image, (28, 28))
        rimg = np.reshape(rimg, (-1,  28*28))
                
        return rimg


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
       
       text = "<b><i>👋Assalomu alaykum!\n\n🤖Digit Recognizer botiga xush kelibsiz.\n\n Botning vazifasi yuborilgan rasmdagi 🔢 raqmni aniqlab berish.</i></b>"
       await context.bot.send_message(chat_id=update.message.chat_id, text=text, parse_mode=ParseMode.HTML)

async def note(update: Update, context: ContextTypes.DEFAULT_TYPE):
        
        text = "<b><i> Eslatma: \n\nSiz bu botga raqm rasmini yuborasiz, bot esa qaysi raqm ekaligini (93 %  to 95 %) deyarli aniqlab beradi.\n\n Yuborilaydigan rasm qo'lda yozilgan(handwritten) raqam bo'lishi kerak!</i></b>"
        
        await context.bot.send_message(chat_id=update.message.chat_id, text=text, parse_mode=ParseMode.HTML)


async def photo_num(update: Update, context: ContextTypes.DEFAULT_TYPE):
       
                
                file_id =  await update.message.photo[-1].get_file()
                file_object = await file_id.download_as_bytearray()
                if file_object:
                      await context.bot.send_message(chat_id=update.message.chat_id, text="<b> <i> Iltimos, biroz kutib turing🕔.!</i></b>", parse_mode=ParseMode.HTML)
                img_path = "image1.jpg"  
                with open(img_path, "wb") as f:
                    f.write(file_object)


                ndata = img2int(img_path)
   
                test_loader  = DataLoader(ndata,  batch_size=700, shuffle=False) 

                with torch.no_grad():
                    for x in test_loader:            
                        x = x.to(device).float()
                        resize = np.resize(x, (28, 28))
                        output = model(x).argmax(dim=1)
                        plt.title(output)
                        plt.imshow(resize)
                        plt.show()
                        
        
                        
                text = f"<b><i>✅Siz yuborgan raqam: {output[0]}</i></b>"
                await context.bot.send_message(chat_id=update.message.chat_id, text=text, parse_mode=ParseMode.HTML)
                
                os.remove(img_path)

application = ApplicationBuilder().token("6208535103:AAGmbP7_Ia9otkjXy3RYsA9vHZzXhjB_AyY").build()

start_handler = CommandHandler("start", start)
application.add_handler(start_handler)

note_handler = CommandHandler("note", note)
application.add_handler(note_handler)

photo_num_handler = MessageHandler(filters.PHOTO, photo_num)
application.add_handler(photo_num_handler)

print("bot ishga tushdi...")
application.run_polling()