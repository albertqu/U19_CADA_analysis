// AUTHOR: Albert Qu, 02/07/2022
#define LASER_PIN 10
#define DT 5  // tone duration in ms
#define FALL_DURATION 250 // LED duration in ms
#define VM 255

String inString = "";
float f_t = 0;
int N_t;
int N_M = floor(FALL_DURATION / DT);
float k = VM / ((float) N_M);

void setup()
{
  Serial.begin(38400);  // initialize serial communications at 9600 bps
  pinMode(LASER_PIN, OUTPUT);
  analogWrite(LASER_PIN, 0);
  N_t = -1;
}
// Cite: [Arduino - StringToIntExample](https://www.arduino.cc/en/Tutorial.StringToIntExample)
void loop()
{ 
  if (Serial.available()>0) { 
    // Available
    int inChar = Serial.read(); // convert the incoming byte to a char and add it to the string:
    if (inChar == '~') {
      // Reward Effected
      analogWrite(LASER_PIN, VM);
      N_t = -1;
      Serial.println(1);
      Serial.println(N_M);
    }
    
    else if (inChar == '!') {
      N_t = 0; // set state in RAMP_OFF
      f_t = VM;
      Serial.println("turning off");
      Serial.println(N_M);
      Serial.println(k);
      Serial.println(VM);
      Serial.println("OFF state");
    }
    
  }
  if (N_t != -1) { // RAMP_OFF state
      Serial.println(f_t);
      Serial.println(N_t);
      if (N_t < N_M) {
        Serial.println("DESC");
        f_t = max(VM - k * N_t, 0);
        N_t += 1;
        Serial.println(f_t);
        Serial.println("updated");
        analogWrite(LASER_PIN, (int) f_t);
      } else {
        Serial.println("FINAL");
        f_t = 0;
        analogWrite(LASER_PIN, f_t);
        N_t = -1;
      }
  }
  delay(DT);
}
