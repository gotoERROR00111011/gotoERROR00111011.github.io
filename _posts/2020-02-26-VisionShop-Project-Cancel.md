---
title: "VisionShop Project (폐기)"

categories:
    - Django

tags :
    - Project

---

## 폐기
프로젝트를 진행하며 Django를 공부하려 하였으나, 아이디어가 구체화 되지 않아 폐기

<br/>

## Start
적당히 주제만 정해둔 프로젝트를 시작한다.    
설계부터 하는것이 맞지만, 나태해지는것을 우려하여 먼저 시작한다.    

## Setup
Windows10 환경    

### 가상환경
기본적으로 [anaconda](https://www.anaconda.com/) 에서 프로젝트를 진행한다.    
```
$ conda create -n vision_shop python=3.6
$ activate vision_shop
```

### Django 설치
```
$ python -m pip install Django
```

### Project 생성
```
$ django-admin startproject mysite
```

### App 생성
```
$ python manage.py startapp VisionShop
```
Project의 urls.py에 App 추가    

### [PostgreSQL](https://www.postgresql.org/)
Django에서 지원되는 DB중 하나인 PostgreSQL을 사용한다.    
```sql
create database visionshop;
create user htlim with password 'comit1234';

alter role htlim set client_encoding to 'utf-8';
alter role htlim set timezone to 'Asia/Seoul';
grant all privileges on database visionshop to htlim;
```
[Project의 setting.py에 DB 설정을 추가한다.](https://docs.djangoproject.com/ko/3.0/ref/databases/#postgresql-notes)    

## Table
```
$ python manage.py migrate
```
### model 정의
models.py에 class로 정의한다.    
### model 활성화
settings.py의 INSTALLED_APPS에 'VisionShop.apps.PollsConfig' 추가    
```
$ python manage.py makemigrations VisionShop
```
### 변경사항 동기화
```
$ python manage.py migrate
```

## View
### view 정의
views.py에 view 추가    
urls.py에 view 연결    
templeate 생성    
(404 ERROR 처리, 단축기능 render 등 참고)    
