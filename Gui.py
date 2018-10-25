import wx
from Ex2 import Bi_rnn

class TabOne(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent)


class TabTwo(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent)

        vbox = wx.BoxSizer(wx.VERTICAL)

        self.l1 = wx.StaticText(self, -1, "Search: ")

        #METHOD:
        hbox1 = wx.BoxSizer(wx.HORIZONTAL)
        self.cb1 = wx.CheckBox(self, label='BI-RNN')
        hbox1.Add(self.cb1)
        self.cb2 = wx.CheckBox(self, label='CNN')
        hbox1.Add(self.cb2, flag=wx.LEFT, border=10)
        self.cb3 = wx.CheckBox(self, label='HAN')
        hbox1.Add(self.cb3, flag=wx.LEFT, border=10)

        #INPUT DOC:
        hbox2 = wx.BoxSizer(wx.HORIZONTAL)
        hbox2.Add(self.l1, 10, wx.EXPAND | wx.ALIGN_LEFT | wx.ALL, 10)
        self.t1 = wx.TextCtrl(self,
                              style=wx.TE_PROCESS_ENTER, size=wx.Size(400, 80))

        hbox2.Add(self.t1, 10, wx.EXPAND | wx.ALIGN_LEFT | wx.ALL, 10)

        #Button
        self.btn = wx.Button(self, -1, "Predict")
        self.btn.Bind(wx.EVT_BUTTON, self.OnClicked)
        hbox2.Add(self.btn, 10, wx.EXPAND | wx.ALIGN_RIGHT | wx.ALL, 10)

        #PREDICTION:
        hbox3 = wx.BoxSizer(wx.HORIZONTAL)
        self.l2 = wx.StaticText(self, -1, "Political Party:   ")
        self.a2 = wx.StaticText(self, -1, "", size=wx.Size(400, 20))
        hbox3.Add(self.l2, 10, wx.EXPAND | wx.ALIGN_LEFT | wx.ALL, 10)
        hbox3.Add(self.a2, 10, wx.EXPAND | wx.ALIGN_LEFT | wx.ALL, 10)

        vbox.Add(hbox1)
        vbox.Add(hbox2)
        vbox.Add(hbox3)

        self.SetSizer(vbox)
        self.Centre()
        self.Show()
        self.Fit()
        self.GetParent()

        print("Loading RNN Model")
        self.bi_rnn = Bi_rnn()
        print("Loaded")


    def OnClicked(self, event):
        print("Enter pressed")
        flag = 0
        if self.cb1.GetValue():
            print("Making RNN Model Prediction")
            results = self.bi_rnn.makePrediction(self.t1.GetValue())
            self.a2.SetLabel(results)
        elif self.cb2.GetValue():
            print("CNN")
        elif self.cb3.GetValue():
            print("HAN")





class TabThree(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent)


class HelloFrame(wx.Frame):
    def __init__(self, parent, title):
        # ensure the parent's __init__ is called
        super(HelloFrame, self).__init__(parent, title = title,size = (700,
                                                                       500));
        panel = wx.Panel(self)
        nb = wx.Notebook(panel)

        # Create the tab windows
        tab1 = TabOne(nb)
        tab2 = TabTwo(nb)
        tab3 = TabThree(nb)

        # Add the windows to tabs and name them.
        nb.AddPage(tab1, "Ad Hoc")
        nb.AddPage(tab2, "Classification")
        nb.AddPage(tab3, "Stats")

        sizer = wx.BoxSizer()
        sizer.Add(nb, 1, wx.EXPAND)
        panel.SetSizer(sizer)

        self.CreateStatusBar()

if __name__ == '__main__':
    app = wx.App()

    frm = HelloFrame(None, 'PRI_Project')

    frm.Show()
    app.MainLoop()