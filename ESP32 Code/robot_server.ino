/*
 * ============================================================
 *  Robot WiFi Command Server
 *  FireBeetle ESP32-S3
 * ============================================================
 */

#include <WiFi.h>

// WiFi 
const char* AP_SSID    = "ROBOT";
const char* AP_PASSWORD = "12345678"; 
const int   PORT          = 8888;  // TCP server port

#define ESC1_PIN  D13  // Right motor
#define ESC2_PIN  D12  // Left motor

// PWM values (µs)
#define PWM_STOP      1500
#define PWM_FWD       1650
#define PWM_REV       1350
#define PWM_TURN_FWD  1650
#define PWM_TURN_REV  1350
#define PWM_MIN       1100
#define PWM_MAX       1900

// TCP server
WiFiServer server(PORT);
WiFiClient client;

// Convert microseconds to 14-bit duty cycle at 50 Hz
void writeUS(uint8_t pin, uint16_t us) {
  // Clamp to safe range before writing
  us = constrain(us, PWM_MIN, PWM_MAX);
  uint32_t duty = (uint32_t)us * (1 << 14) / (1000000 / 50);
  analogWrite(pin, duty);
}

// Motor commands
void motorForward() { writeUS(ESC1_PIN, PWM_FWD);      
                      writeUS(ESC2_PIN, PWM_FWD);}
void motorBack()    { writeUS(ESC1_PIN, PWM_REV);      
                      writeUS(ESC2_PIN, PWM_REV);}
void motorLeft()    { writeUS(ESC1_PIN, PWM_TURN_FWD); 
                      writeUS(ESC2_PIN, PWM_TURN_REV);}
void motorRight()   { writeUS(ESC1_PIN, PWM_TURN_REV); 
                      writeUS(ESC2_PIN, PWM_TURN_FWD);}
void motorStop()    { writeUS(ESC1_PIN, PWM_STOP);     
                      writeUS(ESC2_PIN, PWM_STOP);}

// MOTOR <left_us> <right_us> parser
void motorDirect(uint16_t leftUS, uint16_t rightUS) {
  writeUS(ESC1_PIN, leftUS);
  writeUS(ESC2_PIN, rightUS);
}

// Command dispatcher
void executeCommand(String cmd) {
  cmd.trim();
  cmd.toUpperCase();

  Serial.print("CMD: ");
  Serial.println(cmd);

  if      (cmd == "FORWARD") motorForward();
  else if (cmd == "BACK")    motorBack();
  else if (cmd == "LEFT")    motorLeft();
  else if (cmd == "RIGHT")   motorRight();
  else if (cmd == "STOP")    motorStop();

  // MOTOR <leftUS> <rightUS>
  else if (cmd.startsWith("MOTOR ")) {
    // Strip the "MOTOR " prefix and parse two integers
    String params = cmd.substring(6);   // everything after "MOTOR "
    params.trim();
    int spaceIdx = params.indexOf(' ');

    if (spaceIdx > 0) {
      uint16_t leftUS  = (uint16_t) params.substring(0, spaceIdx).toInt();
      uint16_t rightUS = (uint16_t) params.substring(spaceIdx + 1).toInt();
      motorDirect(leftUS, rightUS);
    } else {
      Serial.println("  MOTOR: bad format — expected MOTOR <left> <right>");
    }
  }

  else {
    Serial.println("  Unknown command — ignored.");
  }
}

// ESC arming
void armESCs() {
  Serial.println("[3] Configuring PWM pins...");
  analogWriteFrequency(ESC1_PIN, 50);
  analogWriteResolution(ESC1_PIN, 14);
  analogWriteFrequency(ESC2_PIN, 50);
  analogWriteResolution(ESC2_PIN, 14);

  Serial.println("[4] Sweeping neutral to arm ESCs...");
  for (int i = 0; i < 5; i++) {
    for (uint16_t us = 1480; us <= 1520; us++) {
      writeUS(ESC1_PIN, us);
      writeUS(ESC2_PIN, us);
      delay(20);
    }
    delay(100);
  }
  motorStop();
  delay(500);
  Serial.println("[5] ESCs armed.");
}

// WiFi
void connectWiFi() {
  WiFi.mode(WIFI_AP);
  WiFi.softAP(AP_SSID, AP_PASSWORD);
  delay(500);
  
  Serial.println("[1] Access Point Started. SSID: " + String(AP_SSID));
  Serial.println("[2] IP address: " + WiFi.softAPIP().toString());
}

// Setup & Loop
void setup() {
  Serial.begin(115200);
  Serial.println("Robot WiFi Command Server — FireBeetle ESP32-S3");
  
  connectWiFi();
  armESCs();
  
  server.begin();
  Serial.print("[6] TCP server on port ");
  Serial.println(PORT);
  Serial.println("Commands: FORWARD, BACK, LEFT, RIGHT, STOP, MOTOR <l> <r>");
}

void loop() {
  if (!client || !client.connected()) {
    client = server.accept();
    if (client) {
      Serial.println("Client connected.");
    }
  }

  if (client && client.connected() && client.available()) {
    String cmd = client.readStringUntil('\n');
    executeCommand(cmd);
    client.println("OK:" + cmd);
  }
}
