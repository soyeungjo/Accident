#%%
import time
import os
import re
import numpy as np
import pandas as pd
from pandas.core.indexes.base import Index

from tqdm import tqdm

#%%
# !pip install chromedriver_autoinstaller
import selenium
from bs4 import BeautifulSoup
import chromedriver_autoinstaller
from selenium import webdriver
from selenium.webdriver import ActionChains

from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By

from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select, WebDriverWait

from selenium.common.exceptions import NoSuchElementException, StaleElementReferenceException, ElementNotInteractableException

from html_table_parser import parser_functions as parser


#%%
url = 'https://www.csi.go.kr/acd/acdCaseList.do'

option = webdriver.ChromeOptions()
option.add_argument('window-size=1920,1080')

driver_path = chromedriver_autoinstaller.install()
driver = webdriver.Chrome(driver_path, options = option)
driver.implicitly_wait(5)
driver.get(url)

#%%
page_path = '//*[@id="main"]/div/div[2]/div/div/div'
page_length_tmp = driver.find_element_by_xpath(page_path).text
page_length = int(page_length_tmp.split(' ')[3])


accid_dict = []
start = time.time()
for j in range(1, page_length):
    accid_num_path = '//*[@id="main"]/div/div[1]/table/tbody'
    accid_num_tmp = driver.find_element_by_xpath(accid_num_path)
    accid_num = len(accid_num_tmp.find_elements_by_tag_name('tr'))

    row_length = tqdm(range(1, accid_num + 1), file = sys.stdout)

    for i in row_length:
        accid_path = '/html/body/div[3]/div[3]/div/div/div[3]/section/form/div/div[1]/table/tbody/tr[%d]/td[2]/a' % i
        accid_url_path = driver.find_element_by_xpath(accid_path)
        accid_url = accid_url_path.get_attribute('href')
        driver.execute_script(accid_url)
        time.sleep(1)
        
        # -------------------------------- 사고사례 -------------------------------- #
        tab = driver.find_elements_by_class_name('table.table-bordered')
        case_tab = tab[0]
        
        not_head = ['사고유형', '사고분류', '사고위치', '피해상황']
        case_head = [x.text for x in case_tab.find_elements_by_class_name('td-head') if x.text not in not_head]
        case_val = [x.text for x in case_tab.find_elements_by_class_name('t-left')]
        
        case_tmp = {case_head[i]: case_val[i] for i in range(len(case_head))}
        
        # -------------------------------- 현장특성 -------------------------------- #
        char_tab = tab[1]
        
        char_head = [x.text for x in char_tab.find_elements_by_class_name('td-head')]
        char_val = [x.text for x in char_tab.find_elements_by_class_name('t-left')]
        
        char_tmp = {char_head[i]: char_val[i] for i in range(len(char_head))}
        case_tmp.update(char_tmp)
        
        accid_dict.append(case_tmp)
        
        driver.back()
        
        row_length.set_postfix({'page': j})
    
    if j % 10 == 0:
        next_page_path = '/html/body/div[3]/div[3]/div/div/div[3]/section/form/div/div[2]/div/ul/li[13]/a'
        driver.find_element_by_xpath(next_page_path).click()
        time.sleep(1)
        
    elif j % 10 != 0: 
        p = j % 10
        page_path = '/html/body/div[3]/div[3]/div/div/div[3]/section/form/div/div[2]/div/ul'
        page_list = driver.find_element_by_xpath(page_path).find_elements_by_tag_name('li')[2:-2]
        page_list[p].find_element_by_tag_name('a').click()
        time.sleep(1)
    
print('\ntime: ', time.time() - start)


#%%
accident = pd.DataFrame(accid_dict)
accident = accident.sort_values('발생일시', ascending = False).reset_index(drop = True)


accident.to_csv('accident.csv', header = True, index = False, encoding = 'utf-8-sig')

accident.head(2)