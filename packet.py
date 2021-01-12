"""
#################################
# Packet Class
#################################
"""

#########################################################
# Class definition


class Packet:

    def __init__(self, arr_time=0, wait_time=-1, depart_time=-1, serv_time=-1, proc_time=-1, status='Gen', user_id=-1,
                 pkt_id=-1, q_id=-1, frme_id=-1):
        """
        :param arr_time: The time stamp that packet is generated.
        :param wait_time: PKT waiting time in the Queue.
        :param depart_time: The time stamp that PKT went out the Queue for relaying.
        :param serv_time: The PKT relaying time by the UAV.
        :param proc_time: The time stamp that PKT is delivered to the BS.
        :param status: Gen=generated, Drop=PKT Dropped, Proc=PKT Serviced.
        :param user_id: PKT is assigned to which primary user.
        :param pkt_id: Index of the PKT for the specific user.
        :param q_id: Index of the PKT in the Queue, not specifically the same as pkt_id.
        :param frme_id: Index of the frame that the PKT was generated.
        """
        self.arrival_time = arr_time
        self.w_time = wait_time
        self.dep_time = depart_time
        self.serv_time = serv_time
        self.proc_time = proc_time
        self.status = status
        self.uid = user_id
        self.pid = pkt_id
        self.qid = q_id
        self.frmeid = frme_id

    def set_arrival(self, arr_time):
        self.arrival_time = arr_time

    def set_wait(self, wait_time):
        self.w_time = wait_time

    def set_depart(self, depart_time):
        self.dep_time = depart_time

    def set_serv(self, serv_time):
        self.serv_time = serv_time

    def set_proc(self, proc_time):
        self.proc_time = proc_time

    def set_status(self, status):
        self.status = status

    def set_userID(self, user_id):
        self.uid = user_id

    def set_pktID(self, pkt_id):
        self.pid = pkt_id

    def set_qID(self, q_id):
        self.qid = q_id

    def set_fID(self, frme_id):
        self.frmeid = frme_id

    def get_arrival(self):
        return self.arrival_time

    def get_qID(self):
        return self.qid

    def get_serv(self):
        return self.serv_time

    def get_pid(self):
        return self.pid

    def get_depart(self):
        return self.dep_time

    def get_status(self):
        return self.status

    def get_fID(self):
        return self.frmeid

    def info_print(self):
        print("arr_time =", self.arrival_time, "\n"
              "wait_time =", self.w_time, "\n"
              "depart_time =", self.dep_time, "\n"
              "serv_time =", self.serv_time, "\n"
              "proc_time =", self.proc_time, "\n"
              "status =", self.status, "\n"
              "user_id =", self.uid, "\n"
              "pkt_id =", self.pid, "\n"
              "q_id =", self.qid, "\n"
              "frame_id =", self.frmeid)

    def service_scheduler(self, t_sim, service_time, atten_factor):
        service_time_tmp = (service_time[self.get_pid()]) / atten_factor
        self.set_serv(service_time_tmp)
        t_sim += service_time_tmp
        process_time = self.get_serv() + self.get_depart()
        self.set_proc(process_time)
        self.set_status("Proc")
        return t_sim
