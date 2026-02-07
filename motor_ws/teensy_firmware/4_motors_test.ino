#include <micro_ros_arduino.h>
#include <stdio.h>
#include <string.h>

// --- COMPATIBILITY PATCH FOR TEENSY 1.58 + MICRO-ROS ---
#include <ctype.h>
#undef __locale_ctype_ptr
extern "C" {
  const unsigned char * __locale_ctype_ptr (void) {
    return (const unsigned char *)_ctype_;
  }
}
// -------------------------------------------------------

// --- LIBRARY CONFLICT FIX ---
#define NONE RCLC_NONE_PLACEHOLDER
#include <rcl/rcl.h>
#include <rcl/error_handling.h>
#include <rclc/rclc.h>
#include <rclc/executor.h>
#undef NONE
// ----------------------------

#include <std_msgs/msg/float64_multi_array.h>
#include <FlexCAN_T4.h>

// --- HARDWARE CONFIG ---
// Teensy 4.0 CAN Pinout: CAN1 = Pin 22 (TX) / Pin 23 (RX)
FlexCAN_T4<CAN1, RX_SIZE_256, TX_SIZE_16> Can0;

// Motor IDs
const int ID_LEFT_1  = 0x01;
const int ID_LEFT_2  = 0x02;
const int ID_RIGHT_1 = 0x03;
const int ID_RIGHT_2 = 0x04;

// --- CONTROL GLOBALS ---
// [Left1, Left2, Right1, Right2]
float target_pos[4] = {0.0, 0.0, 0.0, 0.0};
float target_speed[4] = {5.0, 5.0, 5.0, 5.0};
float actual_pos[4] = {0.0, 0.0, 0.0, 0.0};

bool motors_enabled = false;

// DEBUG GLOBALS
uint32_t can_rx_count = 0;

// --- ROS GLOBALS ---
rcl_subscription_t subscriber;
std_msgs__msg__Float64MultiArray msg_sub;
rcl_publisher_t feedback_pub;
std_msgs__msg__Float64MultiArray msg_feedback;

rclc_executor_t executor;
rclc_support_t support;
rcl_allocator_t allocator;
rcl_node_t node;

// Buffer sizes: 
// Sub: [P1, P2, P3, P4, S1, S2, S3, S4, Mode] = 9 doubles
// Pub: [P1, P2, P3, P4, RX_Count] = 5 doubles
static double msg_data_buffer[9];
static double feedback_data_buffer[5]; 

#define LED_PIN 13

enum states {
  WAITING_AGENT,
  AGENT_AVAILABLE,
  AGENT_CONNECTED,
  AGENT_DISCONNECTED
} state;

// --- ERROR HANDLING ---
#define RCCHECK(fn) { rcl_ret_t temp_rc = fn; if((temp_rc != RCL_RET_OK)){ return false; }}
#define RCSOFTCHECK(fn) { rcl_ret_t temp_rc = fn; (void)temp_rc; }

// --- DAMIAO HELPER FUNCTIONS ---
float uint_to_float(int x_int, float x_min, float x_max, int bits) {
    float span = x_max - x_min;
    float offset = x_min;
    return ((float)x_int) * span / ((float)((1 << bits) - 1)) + offset;
}

void set_motor_state(uint16_t id, bool enable) {
    CAN_message_t msg;
    msg.id = id;
    msg.len = 8;
    for(int i=0; i<7; i++) msg.buf[i] = 0xFF;
    msg.buf[7] = enable ? 0xFC : 0xFD; 
    Can0.write(msg);
}

void pack_and_send_pos_speed(uint8_t id, float p_des, float v_des) {
    CAN_message_t msg;
    msg.id = 0x100 + id; // Offset ID for Pos-Speed Mode
    msg.len = 8;
    memcpy(&msg.buf[0], &p_des, 4);
    memcpy(&msg.buf[4], &v_des, 4);
    Can0.write(msg);
}

// --- READ CAN FEEDBACK ---
void read_can_feedback() {
    CAN_message_t msg;
    while(Can0.read(msg)) {
        can_rx_count++; 

        int id_from_frame = msg.id;
        int id_from_payload = msg.buf[0] & 0x0F; 
        
        uint16_t p_int = ((uint16_t)msg.buf[1] << 8) | (uint16_t)msg.buf[2];
        float p = uint_to_float(p_int, -12.5, 12.5, 16);
        
        int detected_id = -1;
        // Determine ID source
        if (id_from_frame == ID_LEFT_1 || id_from_payload == ID_LEFT_1) detected_id = 0;
        else if (id_from_frame == ID_LEFT_2 || id_from_payload == ID_LEFT_2) detected_id = 1;
        else if (id_from_frame == ID_RIGHT_1 || id_from_payload == ID_RIGHT_1) detected_id = 2;
        else if (id_from_frame == ID_RIGHT_2 || id_from_payload == ID_RIGHT_2) detected_id = 3;

        if (detected_id != -1) {
            actual_pos[detected_id] = p;
        }
    }
}

// --- SUBSCRIPTION CALLBACK ---
void subscription_callback(const void * msin) {
  const std_msgs__msg__Float64MultiArray * msg = (const std_msgs__msg__Float64MultiArray *)msin;
  
  // 1. Position Targets [0-3]
  if (msg->data.size >= 4) {
    for(int i=0; i<4; i++) target_pos[i] = (float)msg->data.data[i];
  }
  
  // 2. Speed Limits [4-7]
  if (msg->data.size >= 8) {
    for(int i=0; i<4; i++) {
        float s = (float)msg->data.data[4+i];
        if (s < 0.5) s = 0.5;
        target_speed[i] = s;
    }
  }

  // 3. Mode (Enable/Disable) [8]
  if (msg->data.size >= 9) {
    float mode = (float)msg->data.data[8];
    bool should_enable = (mode > 0.5);
    
    if (should_enable != motors_enabled) {
        motors_enabled = should_enable;
        if (motors_enabled) {
            // Re-Enable: Sync
            for(int i=0; i<4; i++) target_pos[i] = actual_pos[i];
            
            set_motor_state(ID_LEFT_1, true);
            set_motor_state(ID_LEFT_2, true);
            set_motor_state(ID_RIGHT_1, true);
            set_motor_state(ID_RIGHT_2, true);
        } else {
            set_motor_state(ID_LEFT_1, false);
            set_motor_state(ID_LEFT_2, false);
            set_motor_state(ID_RIGHT_1, false);
            set_motor_state(ID_RIGHT_2, false);
        }
    }
  }
}

// --- ENTITY MANAGEMENT ---
bool create_entities() {
  allocator = rcl_get_default_allocator();
  RCCHECK(rclc_support_init(&support, 0, NULL, &allocator));
  RCCHECK(rclc_node_init_default(&node, "dual_arm_bridge", "", &support));

  // Subscriber
  std_msgs__msg__Float64MultiArray__init(&msg_sub);
  msg_sub.data.capacity = 9;
  msg_sub.data.data = msg_data_buffer;
  msg_sub.data.size = 0;

  RCCHECK(rclc_subscription_init_default(
    &subscriber, &node,
    ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, Float64MultiArray),
    "motor_commands"));

  // Publisher
  std_msgs__msg__Float64MultiArray__init(&msg_feedback);
  msg_feedback.data.capacity = 5; 
  msg_feedback.data.data = feedback_data_buffer;
  msg_feedback.data.size = 0;

  RCCHECK(rclc_publisher_init_default(
    &feedback_pub, &node,
    ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, Float64MultiArray),
    "motor_feedback"));

  RCCHECK(rclc_executor_init(&executor, &support.context, 1, &allocator));
  RCCHECK(rclc_executor_add_subscription(&executor, &subscriber, &msg_sub, &subscription_callback, ON_NEW_DATA));

  return true;
}

void destroy_entities() {
  RCSOFTCHECK(rcl_subscription_fini(&subscriber, &node));
  RCSOFTCHECK(rcl_publisher_fini(&feedback_pub, &node));
  RCSOFTCHECK(rclc_executor_fini(&executor));
  RCSOFTCHECK(rcl_node_fini(&node));
  RCSOFTCHECK(rclc_support_fini(&support));
  
  msg_sub.data.data = NULL; msg_sub.data.capacity = 0;
  msg_feedback.data.data = NULL; msg_feedback.data.capacity = 0;
  std_msgs__msg__Float64MultiArray__fini(&msg_sub);
  std_msgs__msg__Float64MultiArray__fini(&msg_feedback);
}

// --- SETUP ---
void setup() {
  pinMode(LED_PIN, OUTPUT);
  Can0.begin();
  Can0.setBaudRate(1000000);
  Can0.setMBFilter(ACCEPT_ALL); 

  // Initial Enable Sequence (All 4 motors)
  for(int i=0; i<5; i++) {
    set_motor_state(ID_LEFT_1, true);
    set_motor_state(ID_LEFT_2, true);
    set_motor_state(ID_RIGHT_1, true);
    set_motor_state(ID_RIGHT_2, true);
    delay(10);
  }
  motors_enabled = true;

  Serial.begin(115200);
  set_microros_transports();
  state = WAITING_AGENT;
}

unsigned long last_can_send = 0;
unsigned long last_feedback = 0;

void loop() {
  read_can_feedback();

  switch (state) {
    case WAITING_AGENT:
      if (rmw_uros_ping_agent(100, 1) == RMW_RET_OK) {
        state = AGENT_AVAILABLE;
      } else {
        digitalWrite(LED_PIN, !digitalRead(LED_PIN)); // Blink
        
        // Polling Keep-Alive
        if (millis() - last_can_send > 100) {
           last_can_send = millis();
           // Force disable polling to read positions
           set_motor_state(ID_LEFT_1, false);
           set_motor_state(ID_LEFT_2, false);
           set_motor_state(ID_RIGHT_1, false);
           set_motor_state(ID_RIGHT_2, false);
        }
      }
      break;

    case AGENT_AVAILABLE:
      if (create_entities()) {
        state = AGENT_CONNECTED;
        digitalWrite(LED_PIN, HIGH);
      } else {
        state = WAITING_AGENT;
        destroy_entities();
      }
      break;

    case AGENT_CONNECTED:
      if (rclc_executor_spin_some(&executor, RCL_MS_TO_NS(1)) != RCL_RET_OK) {
          state = AGENT_DISCONNECTED;
          return;
      }
      
      // Control Loop (50Hz)
      if (millis() - last_can_send > 20) {
         last_can_send = millis();
         
         if (motors_enabled) {
             pack_and_send_pos_speed(ID_LEFT_1, target_pos[0], target_speed[0]);
             pack_and_send_pos_speed(ID_LEFT_2, target_pos[1], target_speed[1]);
             pack_and_send_pos_speed(ID_RIGHT_1, target_pos[2], target_speed[2]);
             pack_and_send_pos_speed(ID_RIGHT_2, target_pos[3], target_speed[3]);
         } else {
             // Disabled Polling
             set_motor_state(ID_LEFT_1, false);
             set_motor_state(ID_LEFT_2, false);
             set_motor_state(ID_RIGHT_1, false);
             set_motor_state(ID_RIGHT_2, false);
             
             // Sync
             for(int i=0; i<4; i++) target_pos[i] = actual_pos[i];
         }
      }
      
      // Feedback Loop (20Hz)
      if (millis() - last_feedback > 50) {
          last_feedback = millis();
          msg_feedback.data.size = 5;
          for(int i=0; i<4; i++) msg_feedback.data.data[i] = actual_pos[i];
          msg_feedback.data.data[4] = (double)can_rx_count; 
          RCSOFTCHECK(rcl_publish(&feedback_pub, &msg_feedback, NULL));
      }
      break;

    case AGENT_DISCONNECTED:
      destroy_entities();
      state = WAITING_AGENT;
      break;
  }
}