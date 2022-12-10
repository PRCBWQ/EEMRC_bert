class Question(object):
    def getTriggerQ(self):
        return "触发词是什么？"

    def getTypeQ(self, trigger=""):
        if len(trigger) == 0:
            return "事件类型是什么？"
        return "在触发词是 " + trigger + " 的条件下，事件类型是什么？"

    def getRoleQ(self, trigger="", type="", role=""):
        str = role + "是什么?"
        condi = ""
        if len(trigger) != 0:
            condi = "触发词是 " + trigger
        if len(type) != 0:
            condi = condi + "，类型为 " + type
        if len(condi) != 0:
            condi = "在" + condi + " 的条件下,"
        str = condi + str
        return str
