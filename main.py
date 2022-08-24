import telebot
from telebot import types
import torch
import torchvision.models as models
from torchvision import transforms
import pickle
from PIL import Image
import numpy as np


#load label encoder, load model
label_encoder = pickle.load(open("label_encoder.pkl", 'rb'))
model = models.resnet50(pretrained=True)

model.fc = torch.nn.Linear(model.fc.in_features, 120)

#freeze BatchNorm layers
model.eval()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)


model.load_state_dict(torch.load('weights_for_classification.h5', map_location=torch.device('cpu')), strict=False)


def read_file(file_name, mode):
    with open(file_name, mode) as file:
        return file.read()


def predict_one_sample(model, inputs, device='cuda'):
    with torch.no_grad():
        if torch.cuda.is_available():
            inputs = inputs.to(device)
            model.eval()
            logit = model(inputs).cpu()
            probs = torch.nn.functional.softmax(logit, dim=-1).numpy()

        model.eval()
        logit = model(inputs)
        probs = torch.nn.functional.softmax(logit, dim=-1).numpy()
    return probs


def predict(image_path):
    img = Image.open(image_path).convert('RGB')
    img.load()

    user_image_resized = img.resize((224, 224))
    img_val = np.array(user_image_resized)
    img_val = np.array(img_val / 255, dtype='float32')
    img_val = transform(img_val)

    prob_pred = predict_one_sample(model, img_val.unsqueeze(0))
    predicted_proba = np.max(prob_pred) * 100
    y_pred = np.argmax(prob_pred)
    predicted_label = label_encoder.classes_[y_pred]
    predicted_label = predicted_label[:len(predicted_label) // 2] + predicted_label[len(predicted_label) // 2:]

    return predicted_proba, predicted_label



#transforms for image
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


#telebot initialization
path_to_token = 'token_telebot.txt'
bot = telebot.TeleBot(read_file(path_to_token, 'r')) #token is a secret information, save it in file
images_dict = {}


@bot.message_handler(content_types=['text'])
def get_user_message(message):
    if message.text == 'View the list of breeds':
        bot.send_message(message.chat.id, read_file('breeds_list.txt', 'r'))

    else:
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True, row_width=2)
        info_btn = types.KeyboardButton('View the list of breeds')

        markup.add(info_btn)

        bot.send_message(message.chat.id,
                        f"Hello :D\n I am a bot that can determine the breed of the dog in the photo. \
                         Send me a photo and I'll tell you who it shows", reply_markup=markup)
        #bot.register_next_step_handler(message, get_user_photo)



@bot.message_handler(content_types=['photo'])
def get_user_photo(message):
    images_dict[str(message.chat.id)] = []
    try:
        file_info = bot.get_file(message.photo[len(message.photo)-1].file_id)

        downloaded_files = bot.download_file(file_info.file_path)

        src = 'tmp/' + file_info.file_path

        with open(src, 'wb') as new_file:
            new_file.write(downloaded_files)

        bot.reply_to(message, 'The image is being processed...')
        images_dict[str(message.chat.id)].append(src)

    except Exception as e:
        bot.reply_to(message, e)

    if (len(images_dict[str(message.chat.id)]) == 1):
        predicted_proba, predicted_label = predict(images_dict[str(message.chat.id)][-1])
        bot.send_message(message.chat.id, "{} : {:.0f}%".format(predicted_label, predicted_proba))


bot.polling(none_stop=True)
