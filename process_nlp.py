# -*- coding: utf-8 -*-
import os
import json
import nltk
import pymorphy2

def remove_digit(data):
    str2 = ''
    for c in data:
        if c not in ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '«', '»', '–', "\""):
            str2 = str2 + c
    data = str2
    return data


def remove_punctuation(data):
    str2 = ''
    import string
    pattern = string.punctuation
    for c in data:
        if c not in pattern:
            str2 = str2 + c
        else:
            str2 = str2 + ""
    data = str2
    return data


def remove_stopwords(data):
    str2 = ''
    from nltk.corpus import stopwords
    russian_stopwords = stopwords.words("russian")
    for word in data.split():
        if word not in (russian_stopwords):
            str2 = str2 + " " + word
    data = str2
    return data


def remove_short_words(data, length=1):
    str2 = ''
    for line in data.split("\n"):
        str3 = ""
        for word in line.split():
            if len(word) > length:
                str3 += " " + word
        str2 = str2 + "\n" + str3
    data = str2
    return data


def remove_paragraf_to_lower(data):
    data = data.lower()
    data = data.replace('\n', ' ')
    return data


def remove_all(data):
    data = remove_digit(data)
    data = remove_punctuation(data)
    data = remove_stopwords(data)
    data = remove_short_words(data, length=3)
    data = remove_paragraf_to_lower(data)
    return data


def remove_all_mas(data):
    new_data = []
    for item in data:
        new_data.append(remove_all(item))
    return new_data

def get_normal_form_mas(lines):
    import pymorphy3
    morph = pymorphy3.MorphAnalyzer()
    result = []
    for line in lines:
        l=""
        for word in line.split():
            p = morph.parse(word)[0]
            l+=" "+(p.normal_form)
        result.append(l)
    return result


def get_normal_form(words):
    import pymorphy3
    morph = pymorphy3.MorphAnalyzer()
    p = morph.parse(words)[0]
    return p.normal_form


def load_data(filename='report_data.json'):
    with open(filename, "r", encoding="UTF8") as file:
        content = file.read()
    data = json.loads(content)
    return data

def remove_from_patterns(text, pattern):
    str2 = ''
    for c in text:
        if c not in pattern:
            str2 = str2 + c
    return str2

def display(text):
    print(text) 
    print("--------------------------------")

def remove_paragraf_and_toLower(text):
    text = text.lower()
    text = text.replace('\n', ' ')
    text = ' '.join([k for k in text.split(" ") if k])
    return text


def nltk_download():
    nltk.download('stopwords')
    nltk.download('punkt')
    

def calc_intersection_list(list1, list2):
    count = 0
    for item1 in list1:
        for item2 in list2:
            count += calc_intersection_text(item1, item2)
    return count

def calc_intersection_text(text1, text2):
    count = 0
    text1 = str(text1)
    text2 = str(text2)
    for item1 in text1.split():
        for item2 in text2.split():
            if item1 == item2:
                count += 1
    return count


def gen_data():
    data=load_data()
    risk=""
    bdu=""
    for item in data:
        risk+=f"{item['risk']}\t{item['desc']}\n"
        bdu+=f"{item['bdu']}\t{item['desc']} \n"
    with open("risk.txt", "w", encoding="UTF8") as file:
        file.write(risk)
    with open("bdu.txt", "w", encoding="UTF8") as file:
        file.write(bdu)


if __name__ == '__main__':
    # nltk_download()
    gen_data()
