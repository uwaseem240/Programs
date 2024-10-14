#include <IRremote.h>
#include <Servo.h>
#include <stdio.h>
#include "HardwareSerial.h"
#include "ArduinoJson-v6.11.1.h"  //Use ArduinoJson Libraries
 
#define f 16736925  // FORWARD
#define b 16754775  // BACK
#define l 16720605  // LEFT
#define r 16761405  // RIGHT
#define s 16712445  // STOP
 
#define UNKNOWN_F 5316027     // FORWARD
#define UNKNOWN_B 2747854299  // BACK
#define UNKNOWN_L 1386468383  // LEFT
#define UNKNOWN_R 553536955   // RIGHT
#define UNKNOWN_S 3622325019  // STOP
 
#define KEY1 16738455  //Line Teacking mode
#define KEY2 16750695  //Obstacles Avoidance mode
 
#define KEY_STAR 16728765
#define KEY_HASH 16732845
 
/*Arduino pin is connected to the IR Receiver*/
#define RECV_PIN 12
 
/*Arduino pin is connected to the Ultrasonic sensor module*/
#define ECHO_PIN A4
#define TRIG_PIN A5
const int ObstacleDetection = 25;
 
/*Arduino pin is connected to the Motor drive module*/
#define ENA 5
#define ENB 6
#define IN1 7
#define IN2 8
#define IN3 9
#define IN4 11
 
#define LED_Pin 13
 
/*Arduino pin is connected to the IR tracking module*/
#define LineTeacking_Pin_Right 10
#define LineTeacking_Pin_Middle 4
#define LineTeacking_Pin_Left 2
 
#define LineTeacking_Read_Right !digitalRead(10)  //Right
#define LineTeacking_Read_Middle !digitalRead(4)  //Middle
#define LineTeacking_Read_Left !digitalRead(2)    //Left
 
#define carSpeed 250  //PWM(Motor speed/Speed)
 
unsigned int carSpeed_rocker = 250;
#define PIN_Servo 3
Servo servo;              //  Create a DC motor drive object
IRrecv irrecv(RECV_PIN);  //  Create an infrared receive drive object
decode_results results;   //  Create decoding object
 
unsigned long IR_PreMillis;
unsigned long LT_PreMillis;
 
int rightDistance = 0;   //Right distance
int leftDistance = 0;    //left Distance
int middleDistance = 0;  //middle Distance
 
 
int minAngle = 400;//the pulse width, in microseconds, corresponding to the minimum (0-degree) angle on the servo (defaults to 700)
int maxAngle = 2500;
 
String CommandSerialNumber;  //
 
enum SERIAL_mode {
  Serial_rocker,
  Serial_programming,
  Serial_CMD,
} Serial_mode = Serial_programming;
 
enum FUNCTIONMODE {
  IDLE,                  /*free*/
  LineTeacking,          /*Line Teacking Mode*/
  ObstaclesAvoidance,    /*Obstacles Avoidance Mode*/
  Bluetooth,             /*Bluetooth Control Mode*/
  IRremote,              /*Infrared Control Mode*/
}func_mode = LineTeacking;      /*Functional mode*/
 
enum MOTIONMODE {
  LEFT,    /*left*/
  RIGHT,   /*right*/
  FORWARD, /*forward*/
  BACK,    /*back*/
  STOP,    /*stop*/
  LEFT_FORWARD,
  LEFT_BACK,
  RIGHT_FORWARD,
  RIGHT_BACK,
} mov_mode = STOP; /*move mode*/
 
void delays(unsigned long t)
{
  for (unsigned long i = 0; i < t; i++)
  {
    getIRData();      //Infrared Communication Data Acquisition
    delay(2);
  }
}
 
/*ULTRASONIC*/
unsigned int getDistance(void) {  //Getting distance
  static unsigned int tempda = 0;
  unsigned int tempda_x = 0;
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);
  tempda_x = ((unsigned int)pulseIn(ECHO_PIN, HIGH) / 58);
  //tempda = tempda_x;
  if (tempda_x > 150) {
    tempda_x = 150;
  }
  // return tempda;
  return tempda_x;
}
/*
  Control motor：Car movement forward
*/
void forward(bool debug, int16_t in_carSpeed) {
  analogWrite(ENA, in_carSpeed);
  analogWrite(ENB, in_carSpeed);
  digitalWrite(IN1, HIGH);
  digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, HIGH);
  if (debug)
    Serial.println("Go forward!");
}
/*
  Control motor：Car moving backwards
*/
void back(bool debug, int16_t in_carSpeed) {
  analogWrite(ENA, in_carSpeed);
  analogWrite(ENB, in_carSpeed);
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, HIGH);
  digitalWrite(IN3, HIGH);
  digitalWrite(IN4, LOW);
  if (debug)
    Serial.println("Go back!");
}
/*
  Control motor：The car turns left and moves forward
*/
void left(bool debug, int16_t in_carSpeed) {
 
  analogWrite(ENA, in_carSpeed);
  analogWrite(ENB, in_carSpeed);
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, HIGH);
 
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, HIGH);
  if (debug)
    Serial.println("Go left!");
}
/*
  Control motor：The car turns right and moves forward
*/
void right(bool debug, int16_t in_carSpeed) {
  analogWrite(ENA, in_carSpeed);
  analogWrite(ENB, in_carSpeed);
  digitalWrite(IN1, HIGH);
  digitalWrite(IN2, LOW);
  digitalWrite(IN3, HIGH);
  digitalWrite(IN4, LOW);
  if (debug)
    Serial.println("Go right!");
}
 
void forward_left(bool debug, int16_t in_carSpeed) {
  analogWrite(ENA, in_carSpeed / 2);
  analogWrite(ENB, in_carSpeed);
  digitalWrite(IN1, HIGH);
  digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, HIGH);
  if (debug)
    Serial.println("Go right!");
}
 
void forward_right(bool debug, int16_t in_carSpeed) {
  analogWrite(ENA, in_carSpeed);
  analogWrite(ENB, in_carSpeed / 2);
  digitalWrite(IN1, HIGH);
  digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, HIGH);
  if (debug)
    Serial.println("Go right!");
}
 
void back_left(bool debug, int16_t in_carSpeed) {
  analogWrite(ENA, in_carSpeed / 2);
  analogWrite(ENB, in_carSpeed);
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, HIGH);
  digitalWrite(IN3, HIGH);
  digitalWrite(IN4, LOW);
  if (debug)
    Serial.println("Go right!");
}
 
void back_right(bool debug, int16_t in_carSpeed) {
  analogWrite(ENA, in_carSpeed);
  analogWrite(ENB, in_carSpeed / 2);
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, HIGH);
  digitalWrite(IN3, HIGH);
  digitalWrite(IN4, LOW);
  if (debug)
    Serial.println("Go right!");
}
 
/*
  Stop motor control：Turn off the motor drive
*/
void stop(bool debug = false) {
  digitalWrite(ENA, LOW);
  digitalWrite(ENB, LOW);
  if (debug)
    Serial.println("Stop!");
}
/*
 Servo Control angle Setting
*/
void ServoControl(uint8_t angleSetting)
{
  servo.attach(3,minAngle,maxAngle);
  servo.write(angleSetting);  //sets the servo position according to the  value
  delay(500);
}
 
 
/*Line Teacking Mode*/
void line_teacking_mode(void) {
  if (func_mode == LineTeacking) {
    if (LineTeacking_Read_Middle) {  //Detecting in the middle infrared tube
 
      forward(false, 110);  //Control motor：the car moving forward
      LT_PreMillis = millis();
    } else if (LineTeacking_Read_Right) {  //Detecting in the right infrared tube
 
      right(false, 120);  //Control motor：the car moving right
      while (LineTeacking_Read_Right) {
 
      }
      LT_PreMillis = millis();
    } else if (LineTeacking_Read_Left) {  //Detecting in the left infrared tube
      left(false, 115);                   //Control motor：the car moving left
      while (LineTeacking_Read_Left) {
        getIRData();  //Infrared data acquisition
      }
      LT_PreMillis = millis();
    } else if(LineTeacking_Read_Left && LineTeacking_Read_Right && LineTeacking_Read_Middle){
        func_mode = ObstaclesAvoidance;
    }else {
      if (millis() - LT_PreMillis > 20) {
        stop();  //Stop motor control：Turn off motor drive mode
        delay(2);
        func_mode = ObstaclesAvoidance;
      }
    }
  }
}
 
 
/*Obstacle avoidance*/
void obstacles_avoidance_mode(void) {
  static boolean first_is = true;
  uint8_t switc_ctrl = 0;
  if (func_mode == ObstaclesAvoidance) {
    if (first_is == true)  //Enter the mode for the first time, and modulate the steering gear to 90 degrees
    {
      ServoControl(90);
      first_is = false;
    }
    uint8_t get_Distance = getDistance();
    if (function_xxx(get_Distance, 0, 15)) {
      stop();
      /*
      ------------------------------------------------------------------------------------------------------
      ServoControl(30 * 1): 0 1 0 1 0 1 0 1
      ServoControl(30 * 3): 0 0 1 1 0 0 1 1
      ServoControl(30 * 5): 0 0 0 0 1 1 1 1
      1 2 4 >>>             0 1 2 3 4 5 6 7 
      1 3 5 >>>             0 1 3 4 5 6 5 9  
      ------------------------------------------------------------------------------------------------------
      Truth table of obstacle avoidance state
      */
      for (int i = 0; i <= 6; i += 3)  //1、3、5 Omnidirectional detection of obstacle avoidance status
      {
        ServoControl(30 * i);
        get_Distance = getDistance();
        delays(200);
        if (function_xxx(get_Distance, 0, 3)) {
          switc_ctrl = 10;
          break;
        } else if (function_xxx(get_Distance, 0, 15))  //How many cm in the front have obstacles?
        {
          switc_ctrl += i;
        }
      }
      ServoControl(90);
    } else  //if (function_xxx(get_Distance, 20, 50))
    {
      forward(false, 100);  //Control car forwar
    }
    while (switc_ctrl) {
      switch (switc_ctrl) {
        case 1:
        case 5:
        case 6:
          forward(false, 65);  //Control car forwar
          switc_ctrl = 0;
          break;
        case 3:
          left(false, 170);  //Control car left
          switc_ctrl = 0;
          break;
        case 4:
          left(false, 170);  //Control car left
          switc_ctrl = 0;
          break;
        case 8:
        case 11:
          right(false, 170);  //Control car right
          switc_ctrl = 0;
          break;
        case 9:
        case 10:
          back(false, 65);  //Control car Car backwards
          switc_ctrl = 4;
          break;
      }
      ServoControl(90);
    }
  } else {
    first_is = true;
  }
}
 
 
/*f(x) int */
static boolean function_xxx(long xd, long sd, long ed)
{
  if (sd <= xd && xd <= ed)
    return true;
  else
    return false;
}
 
 
/*
  Infrared Communication Data Acquisition
*/
void getIRData(void)
{
  if (irrecv.decode(&results))
  {
    IR_PreMillis = millis();
    switch (results.value)
    {
    case f:
    case UNKNOWN_F:
      func_mode = IRremote;
      mov_mode = FORWARD;
      break; /*forward*/
    case b:
    case UNKNOWN_B:
      func_mode = IRremote;
      mov_mode = BACK;
      break; /*back*/
    case l:
    case UNKNOWN_L:
      func_mode = IRremote;
      mov_mode = LEFT;
      break; /*left*/
    case r:
    case UNKNOWN_R:
      func_mode = IRremote;
      mov_mode = RIGHT;
      break; /*right*/
    case s:
    case UNKNOWN_S:
      func_mode = IRremote;
      mov_mode = STOP;
      break; /*stop*/
    case KEY1:
      func_mode = LineTeacking;
      break; /*Line Teacking Mode*/
    case KEY2:
      func_mode = ObstaclesAvoidance;
      break; /*Obstacles Avoidance Mode*/
    default:
      break;
    }
    irrecv.resume();
  }
}
 
void setup(void) {
  Serial.begin(9600);  //initialization
  ServoControl(90);
  irrecv.enableIRIn();  //Enable infrared communication NEC
 
  pinMode(ECHO_PIN, INPUT);  //Ultrasonic module initialization
  pinMode(TRIG_PIN, OUTPUT);
 
  pinMode(IN1, OUTPUT);  //Motor-driven port configuration
  pinMode(IN2, OUTPUT);
  pinMode(IN3, OUTPUT);
  pinMode(IN4, OUTPUT);
  pinMode(ENA, OUTPUT);
  pinMode(ENB, OUTPUT);
 
  pinMode(LineTeacking_Pin_Right, INPUT);  //Infrared tracking module port configuration
  pinMode(LineTeacking_Pin_Middle, INPUT);
  pinMode(LineTeacking_Pin_Left, INPUT);
}
 
void loop(void) {
  line_teacking_mode();        //Line Teacking Mode
  obstacles_avoidance_mode();  //Obstacles Avoidance Mode
}
 
