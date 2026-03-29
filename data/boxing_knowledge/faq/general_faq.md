# Boxing Training FAQ

Frequently asked questions about boxing training, technique, and the BoxBunny system.

---

## Getting Started

### Q: What size boxing gloves should I get?
**A**: For bag work, 12 oz gloves are standard for lighter individuals (under 65 kg) and 14 oz for heavier individuals. For sparring, always use 16 oz gloves regardless of weight. The extra padding protects both you and your partner. When using the BoxBunny system, 12-14 oz gloves work well for all drill modes.

### Q: Do I need to wrap my hands?
**A**: Yes, always. Hand wraps protect the small bones in the hand and wrist, and they keep the glove snug against the fist. Proper wrapping supports the wrist joint and cushions the knuckles. Use 180-inch (4.5m) Mexican-style wraps for the best protection.

### Q: How often should I train as a beginner?
**A**: Start with 3 sessions per week, with at least one rest day between sessions. This gives your body time to recover and adapt. After 4-6 weeks, you can increase to 4 sessions per week. Overtraining as a beginner leads to injury and burnout.

### Q: Is boxing safe for beginners?
**A**: Bag work, pad work, shadow boxing, and BoxBunny training are very safe. The injury risk comes primarily from sparring (person-to-person). As a beginner, you should spend months developing fundamentals before engaging in any sparring. The BoxBunny system provides a safe training environment with no person-to-person contact.

### Q: I'm not fit enough to start boxing. Should I get in shape first?
**A**: No. Start now. Boxing itself will get you in shape. Trying to "get fit before boxing" is a common trap that delays starting indefinitely. Begin at your own pace, take breaks when needed, and your fitness will improve rapidly.

---

## Technique

### Q: Orthodox or southpaw -- which should I choose?
**A**: If you are right-handed, use orthodox stance (left foot forward, right hand in the rear). If you are left-handed, use southpaw (right foot forward, left hand in the rear). Your dominant hand becomes your power hand (rear hand). Some coaches experiment with switching, but start with the conventional choice.

### Q: Why does my wrist hurt when I hit the heavy bag?
**A**: This usually means your wrist is not straight at impact, your wraps are loose, or your fist is not properly closed. At impact, the fist, wrist, and forearm must form a perfectly straight line. Tighten your wraps, clench the fist before impact, and start with lighter punches until the alignment becomes natural.

### Q: How do I know if I'm punching correctly?
**A**: Key indicators of correct punching: (1) you feel the impact in your shoulder and core, not just your arm; (2) the bag compresses rather than swings wildly; (3) you can throw 50 punches without joint pain; (4) your fist returns to your chin automatically. The BoxBunny CV system tracks punch form and provides feedback through the AI coach.

### Q: Should I fully extend my arm when punching?
**A**: Yes, but do not lock the elbow. Full extension means the arm is almost straight but retains a micro-bend at the elbow. Locking the elbow hyperextends the joint and can cause injury. The snap at full extension is what generates the final acceleration.

### Q: How do I stop flinching when punches come at my face?
**A**: Flinching is a natural reflex that diminishes with exposure. Start with slow, controlled drills: have a partner throw very light jabs at 20% speed while you practice keeping your eyes open and blocking. Gradually increase speed over weeks. The BoxBunny defence drill provides a machine-consistent stimulus that helps desensitize the flinch response.

---

## Training

### Q: How long should a boxing round last?
**A**: Professional boxing rounds are 3 minutes with 1 minute of rest. For beginners, 2-minute rounds with 1-minute rest is a good starting point. For conditioning, you can vary the format: 3 minutes work / 30 seconds rest is more demanding; 2 minutes work / 1 minute rest allows more recovery between rounds.

### Q: How many rounds should I do per session?
**A**: Beginners: 3-4 rounds. Intermediate: 6-8 rounds. Advanced: 10-12 rounds. This includes all types of rounds (shadow boxing, bag work, drills, conditioning). Quality always beats quantity -- it is better to do 4 focused rounds than 8 sloppy ones.

### Q: What should I do on rest days?
**A**: Active recovery is best: light walking, stretching, yoga, or swimming. Avoid complete inactivity, as light movement helps flush metabolic waste from the muscles. Do not hit the heavy bag on rest days -- your hands, wrists, and shoulders need the break.

### Q: How long before I see improvement?
**A**: You will feel more comfortable in the stance and with the jab within 2-3 sessions. Noticeable technique improvement typically takes 4-6 weeks of consistent training. Significant fitness gains take 6-8 weeks. True competence (flowing combinations, defensive reflexes) takes 6-12 months.

### Q: Is shadow boxing really important, or can I just hit the bag?
**A**: Shadow boxing is essential. It is where you develop technique without the feedback distortion of the bag. On the bag, many people develop bad habits (pushing instead of snapping, standing still, ignoring footwork) because the bag rewards force rather than form. Shadow box at least 2 rounds per session.

---

## Fitness and Nutrition

### Q: Will boxing help me lose weight?
**A**: Yes. Boxing training burns 500-800 calories per hour depending on intensity. It combines cardiovascular conditioning with full-body muscle engagement. Combined with reasonable nutrition, boxing is one of the most effective activities for body composition improvement.

### Q: What should I eat before training?
**A**: Eat a light meal 1.5-2 hours before training: complex carbohydrates (oats, rice, sweet potato) with a moderate amount of protein (chicken, eggs, yogurt). Avoid heavy, fatty meals that slow digestion. If you train early morning, a banana and a glass of water 30 minutes before is sufficient.

### Q: How much water should I drink?
**A**: Drink 500ml of water in the hour before training. During training, sip water between rounds -- do not gulp large amounts. After training, drink 500ml-1L to replenish. If your urine is dark yellow, you are under-hydrated.

### Q: Do I need supplements for boxing?
**A**: For most people, no. A balanced diet provides everything you need. If you train intensely 5+ times per week, a protein supplement can help with recovery. Creatine can support explosive power but is not necessary for beginners. Consult a sports nutritionist for personalised advice.

---

## BoxBunny System

### Q: What is the difference between the three drill modes?
**A**: (1) **Reaction Time Drill** tests how quickly you can respond to visual stimuli -- it measures your reaction speed. (2) **Shadow Sparring Drill** displays combinations on screen and uses the CV model to verify that you throw them correctly. (3) **Defence Drill** has the robot arm throw punches at you, and the system evaluates your blocking and slipping.

### Q: How does the BoxBunny system detect my punches?
**A**: The system uses a combination of computer vision (Intel D435i depth camera) and IMU sensors in the pads. The CV model classifies the type of punch you throw based on the motion pattern, and the pad IMU confirms impact and measures force. The two signals are fused together for reliable detection.

### Q: What does the AI coach feedback mean?
**A**: After each session, the AI coach (powered by a local LLM) analyses your performance data -- punch accuracy, combination completion, reaction times, defence rate, and more. It provides specific, actionable feedback: things you did well, areas to improve, and drills to practise. The feedback is tailored to your skill level.

### Q: Can I use BoxBunny without an internet connection?
**A**: Yes. The BoxBunny system runs entirely locally on the Jetson Orin. The LLM model, CV model, and all processing happen on-device. No internet connection is required for any functionality. The phone dashboard connects via local Wi-Fi.

### Q: How accurate is the punch detection?
**A**: The fused CV+IMU system achieves high accuracy for the six standard punch types (jab, cross, left hook, right hook, left uppercut, right uppercut) and blocking. The system is most accurate when the user is within the recommended distance range (1.0-2.5 metres from the camera) and the lighting is adequate.

---

## Safety

### Q: What should I do if my knuckles are bruised?
**A**: Ice them for 10-15 minutes after training. Ensure your hand wraps are providing adequate knuckle coverage. If bruising is recurrent, add gel knuckle guards underneath the wraps. If bruising is severe or accompanied by swelling, rest until fully healed before hitting the bag again.

### Q: How do I prevent shoulder injuries?
**A**: Warm up the shoulders thoroughly before every session (arm circles, band pull-aparts). Strengthen the rotator cuff with external rotation exercises. Do not overtrain -- shoulder fatigue increases injury risk. If you feel pain (not soreness), stop and rest.

### Q: Is it normal to feel sore after boxing?
**A**: Yes, especially in the first few weeks. Muscle soreness (DOMS) in the shoulders, arms, core, and legs is normal. Sharp joint pain, persistent wrist pain, or numbness is not normal and should be evaluated. Light training through mild soreness is fine; training through pain is not.
