Soo whast's 

        This was a compitision code and the comptition is 2Coool kaggle compition it was held on Aug-24-2025 to Sep-29-2025.

The codes and repro

        So, first we have a file named Accident_model_treaning.py it was a traning code for our accident classifire model you have to run the code on any gpu like T4 or P100 any . then it will give you the output accident_model.pth Note : if you dont wannat to train the model you can download the model form kaggel -- https://www.kaggle.com/models/haradibots/1st-model_accident_classifire/PyTorch/default/1

        next we have pridictor code for the accident classifire model Testing_of_Accident_pth.py you can test the accident model from this code it will give you the output lookin like {0: "No Incident", 1: "Near Collision", 2: "Collision"} 

        next we have yolo counter code Yolo_model_Counter.py  it was a code that can take the input a video then break in freams then prosses the freams around 16-24 freams then it use yolov8m to count and classification of object then it have a future of traking it can treak the object/car/animal whre it was going e.g. a animal ↑ moving away
        and it gives output in many ways (ojbect counts) , (movement narrative) and last (output.mp4) 

        Example (object counts):
                Final Video Summary:
                car: 5
                person: 3
                bus: 1
                dog: 1

        Example (movement narrative):
                Traffic Narrative:
                Ego car is driving in traffic.
                Detected 5 car(s), with movements: 3 ↓ approaching, 2 → right.
                Detected 3 person(s), with movements: 2 ↑ moving away, 1 ← left.
                Detected 1 bus(s).
                Detected 1 dog(s).

        Example (output.mp4):
               https://www.dropbox.com/scl/fi/3qxxrugjyrkqj50owvlgf/output.mp4?rlkey=zn8khq90iqkt2q5inb8gfyq0h&st=9yqycqvt&dl=0


Last code is about 
       It was the main code for my Compitision so i dont wanna givee it to any one but here is the code 
       so, we have file named (Both_mix_code_submition.py) so it was a combination of my all models and codes it was code that take the video clip input then gives the output like this : 

       video,Incident window start frame,Incident Detection,Crash Severity,Ego-car involved,Label,Number of Bicyclists/Scooters,Number of animals involved,Number of pedestrians involved,Number of vehicles involved (excluding ego-car),Caption Before Incident,Reason of Incident
        558,390,Near Collision,4. Other cars collided but ego-car is ok,9,multi-vehicle collision (ego not involved),0,2,399,6,Ego-car is driving in heavy traffic.,Other vehicles collided near ego-car.
        489,0,No incident,0. No accident,0,no incident,0,0,1,7,Ego-car is driving in heavy traffic.,No accident occurred.
        669,0,No incident,0. No accident,0,no incident,0,0,0,2,Ego-car is driving with light traffic.,No accident occurred.

        i dont wanna tell more about this code Tq so much 
         build by Aditya(HaradiBots) with love. 

         cheak our website haradibots.onrender.com 
         or msg on whatsapp:7887285338
         insta : llaka2937
         