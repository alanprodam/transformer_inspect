import json
from json import JSONDecodeError
from VisualInspection import Inspection
# from time import sleep
import time

debug = True


class ClassifierExecutionWrapper:
    def __init__(self):
        self.inspection = Inspection()
        self.cycleCount = 0
        self.sampleCount = 0
        self.detection = False
        self.listUpside = []
        self.listDownside = []

    def receiveProtocol(self, message):
        try:
            data = json.loads(message)
            identifier = data["id"].split("@")[0]
            return identifier, data
        except JSONDecodeError:
            return False, {"action": "unknown"}
        except KeyError:
            return False, {"action": "protocol needs an identifier"}

    def run(self):
        try:
            while True:
                identifier, receiveCMD = self.receiveProtocol(input())
                if receiveCMD["action"] == "start_cycle":
                    if len(self.listUpside) == 0 and len(self.listDownside) == 0 and self.detection == False:
                        self.cycleCount += 1
                        self.sampleCount = 0
                        self.detection = True

                        if debug:
                            print('start_cycle...')
                            print({
                                'cycle_count': self.cycleCount,
                                'sample': self.sampleCount,
                                'detection': self.detection
                            })

                        try:
                            result = {"detection": "1"}
                            print(json.dumps(result))
                        except KeyboardInterrupt:
                            result = {"detection": "0"}
                            print(json.dumps(result))
                    else:
                        self.cycleCount = 0
                        self.sampleCount = 0
                        self.listUpside = []
                        self.listDownside = []
                        self.cycleCount += 1
                        self.detection = True

                        if debug:
                            print('start_cycle...')
                            print({
                                'cycle_count': self.cycleCount,
                                'sample': self.sampleCount,
                                'detection': self.detection
                            })

                        try:
                            result = {"detection": "1"}
                            print(json.dumps(result))
                        except KeyboardInterrupt:
                            result = {"detection": "0"}
                            print(json.dumps(result))

                elif receiveCMD["action"] == "test":
                    try:
                        self.sampleCount += 1
                        if debug:
                            print('-----')
                            print('Triger test...')
                            print({
                                'cycle_count': self.cycleCount,
                                'sample': self.sampleCount,
                                'detection': self.detection
                            })
                        if self.sampleCount <= 8:
                            try:
                                # sampleLeft, sampleRight = self.inspection.callInspect(self.sampleCount, self.detection)
                                self.inspection.transformerDetection()
                                sampleLeft = 1
                                sampleRight = 1
                                # time.sleep(2)
                                if 1 <= self.sampleCount <= 4:
                                    self.listUpside.append(sampleLeft)
                                    self.listUpside.append(sampleRight)
                                    if debug:
                                        print('listUpside: ', self.listUpside)
                                elif 5 <= self.sampleCount <= 8:
                                    self.listDownside.append(sampleLeft)
                                    self.listDownside.append(sampleRight)
                                    if debug:
                                        print('listDownside: ', self.listDownside)

                                print(json.dumps({"detection": "1"}))
                                if debug:
                                    print('-----')
                            except KeyboardInterrupt:
                                result = {"detection": "0"}
                                print(json.dumps(result))
                        else:
                            result = {"detection": "0"}
                            if debug:
                                print('Ciclo não encerrado!')
                            print(json.dumps(result))
                    except Exception:
                        print(json.dumps({"response": "cant execute test"}))
                elif receiveCMD["action"] == "end_cycle":
                    self.detection = False
                    if debug:
                        print('end_cycle')
                        print({
                            'cycle_count': self.cycleCount,
                            'sample': self.sampleCount,
                            'detection': self.detection
                        })

                    try:
                        result = {
                            "upside": self.listUpside,
                            "downside": self.listDownside
                        }

                        if len(self.listUpside) != 0 and len(self.listDownside) != 0:
                            print(json.dumps(result))
                            self.listUpside = []
                            self.listDownside = []
                        else:
                            if debug:
                                print('Ciclo encerrado, inicie um novo ciclo!')
                            result = {"detection": "0"}
                            print(json.dumps(result))
                    except KeyboardInterrupt:
                        result = {"detection": "0"}
                        print(json.dumps(result))
                elif receiveCMD["action"] == "ping":
                    print(json.dumps({
                        "id": identifier,
                        "action": "ping",
                        "response": "pong"
                    }))
                else:
                    print(json.dumps({
                        "action": "unknown"
                    }))
        except KeyboardInterrupt:
            # Received interruption
            pass


if __name__ == "__main__":
    ClassifierExecutionWrapper().run()