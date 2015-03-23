################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../batch.cpp \
../bibnumber.cpp \
../facedetection.cpp \
../log.cpp \
../pipeline.cpp \
../textdetection.cpp \
../textrecognition.cpp \
../train.cpp 

OBJS += \
./batch.o \
./bibnumber.o \
./facedetection.o \
./log.o \
./pipeline.o \
./textdetection.o \
./textrecognition.o \
./train.o 

CPP_DEPS += \
./batch.d \
./bibnumber.d \
./facedetection.d \
./log.d \
./pipeline.d \
./textdetection.d \
./textrecognition.d \
./train.d 


# Each subdirectory must supply rules for building sources it contributes
%.o: ../%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: Cross G++ Compiler'
	g++ -O0 -g3 -pedantic -Wall -Wextra -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


