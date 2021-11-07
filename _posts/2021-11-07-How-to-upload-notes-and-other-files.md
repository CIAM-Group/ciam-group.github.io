---
layout: post
title: 如何上传笔记及其他文件
date: 2021-10-04
category: "Others"
author: LiHan
tags: [blog维护]
---

## 文件目录结构说明

<img src="{{ '/assets/imgs/How-to-upload-notes-and-other-files/1.png' | relative_url }}" style="zoom:100%;">

## 上传文件

<img src="{{ '/assets/imgs/How-to-upload-notes-and-other-files/2.png' | relative_url }}" style="zoom:100%;">

<img src="{{ '/assets/imgs/How-to-upload-notes-and-other-files/3.png' | relative_url }}" style="zoom:100%;">

## 上传笔记

<img src="{{ '/assets/imgs/How-to-upload-notes-and-other-files/4.png' | relative_url }}" style="zoom:100%;">

<img src="{{ '/assets/imgs/How-to-upload-notes-and-other-files/5.png' | relative_url }}" style="zoom:100%;">

## 提示

1. 做出修改之后，GitHub page需要一小段时间才会更新，多刷新几次；
2. 搭建过程使用了GitHub page + Jekyll，本文没有对其他文件进行说明。因为如果仅做上传操作，不需要了解其他，如果想要自定义更多内容 (如在blog中增加自定义 js/css)，请查看Jekyll官方文档了解其他文件；
3. 上文展示的是简易版本，当然也可以选择git  clone到本地之后做修改然后push，需要修改的文件位置，格式规定等与简易版本相同；如果clone到本地，可以用`jekyll serve`在本地预览后再push；
4. git上传的忽略规则在根目录下的`.gitignore`，忽略一些不建议上传的文件 (如MacOS下产生的`.DS_Store`) 。如果有另外需要忽视的文件，请更新此规则。

