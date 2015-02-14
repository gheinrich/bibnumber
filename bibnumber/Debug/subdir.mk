################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../TextDetection.cpp \
../batch.cpp \
../bibnumber.cpp 

OBJS += \
./TextDetection.o \
./batch.o \
./bibnumber.o 

CPP_DEPS += \
./TextDetection.d \
./batch.d \
./bibnumber.d 


# Each subdirectory must supply rules for building sources it contributes
%.o: ../%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: Cross G++ Compiler'
	g++ -I/home/greg/ws/boost_1_57_0 -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


