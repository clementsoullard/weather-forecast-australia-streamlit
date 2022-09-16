#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 10:04:22 2022

@author: yohancohen
"""

# Core Pkg
import streamlit as st

# Custom modules
from streamlit_base import bases_streamlit # Basic streamlit function
from demo_australie import demo_australie # Basic ML web app with stremlit


def main():

    # List of pages
   # liste_menu = ["bases streamlit", "demo_ML"]

    # Sidebar
  #  menu = st.sidebar.selectbox("selectionner votre activité", liste_menu)

    # Page navigation
 #   if menu == liste_menu[1]:
 #       bases_streamlit()
 #   else:
    demo_australie()


if __name__ == '__main__':
    main()
