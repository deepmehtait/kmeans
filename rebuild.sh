#!/bin/bash

dirPath="/user/user01/LAB3_SUBMISSION/E1"
mkdir $dirPath/classes
javac -d $dirPath/classes $dirPath/KmeansAlgo.java
jar -cvf kmeans.jar -C $dirPath/classes/ .
mv kmeans.jar $dirPath/

