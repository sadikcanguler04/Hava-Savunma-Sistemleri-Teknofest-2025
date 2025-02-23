#include <Servo.h>

Servo pan_servo;
Servo tilt_servo;

int pan_angle1 = 90;  // Başlangıç açısı
int tilt_angle1 = 90; // Başlangıç açısı
int pan_angle = 90;  // Başlangıç açısı
int tilt_angle = 90; // Başlangıç açısı


void setup() {
  // Serilerle iletişimi başlat
 Serial.begin(115200);  
  // Pan ve tilt servolarını pinlere bağla
  pan_servo.attach(5);  // Pan servo için pin 9
  tilt_servo.attach(6); // Tilt servo için pin 10
  pinMode(3, OUTPUT);  // turn the LED on (HIGH is the voltage level)

  digitalWrite(3, HIGH);  // turn the LED on (HIGH is the voltage level)
  // Başlangıç pozisyonlarına git
  pan_servo.write(pan_angle1);
  tilt_servo.write(tilt_angle1);
  
  delay(2000); // Başlangıçta servoların hareket etmesi için 2 saniye bekle
}

void loop() {
  if (Serial.available() > 0) {
    // Arduino'ya gelen veriyi oku
    String data = Serial.readStringUntil('\n');
    
    // Pan ve tilt açılarını veriden ayır
    int separatorIndex = data.indexOf(',');
    if (separatorIndex != -1) {
      String pan_str = data.substring(0, separatorIndex);
      String tilt_str = data.substring(separatorIndex + 1);
      
      // Açıları integer olarak al
      pan_angle = pan_str.toInt();
      tilt_angle = tilt_str.toInt();
      
      // Açıları servolara gönder
      pan_servo.write(pan_angle);
      tilt_servo.write((tilt_angle));
      
      // Verilen açıları seri monitöre yazdır
      Serial.print("Pan: ");
      Serial.print(pan_angle);
      Serial.print(", Tilt: ");
      Serial.println(tilt_angle);
    }
  }
}
