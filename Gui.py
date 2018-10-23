import wx

class TabOne(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent)


class TabTwo(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent)

        vbox = wx.BoxSizer(wx.VERTICAL)
        hbox1 = wx.BoxSizer(wx.HORIZONTAL)

        self.l1 = wx.StaticText(self, -1, "Search: ")

        hbox1.Add(self.l1, 10, wx.EXPAND | wx.ALIGN_LEFT | wx.ALL, 10)
        self.t1 = wx.TextCtrl(self,
                              style=wx.TE_PROCESS_ENTER, size=wx.Size(400, 80))

        hbox1.Add(self.t1, 10, wx.EXPAND | wx.ALIGN_LEFT | wx.ALL, 10)
        self.t1.Bind(wx.EVT_TEXT_ENTER, self.OnEnterPressed)

        hbox2 = wx.BoxSizer(wx.HORIZONTAL)
        self.l2 = wx.StaticText(self, -1, "Political Party:   ")
        self.a2 = wx.StaticText(self, -1, "", size=wx.Size(400, 20))
        hbox2.Add(self.l2, 10, wx.EXPAND | wx.ALIGN_LEFT | wx.ALL, 10)
        hbox2.Add(self.a2, 10, wx.EXPAND | wx.ALIGN_LEFT | wx.ALL, 10)

        vbox.Add(hbox1)
        vbox.Add(hbox2)
        self.SetSizer(vbox)
        self.Centre()
        self.Show()
        self.Fit()

    def OnEnterPressed(self, event):
        print("Enter pressed")


class TabThree(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent)


class HelloFrame(wx.Frame):
    def __init__(self, parent, title):
        # ensure the parent's __init__ is called
        super(HelloFrame, self).__init__(parent, title = title,size = (500,
                                                                       250));
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

    def OnEnterPressed(self, event):
        print("Enter pressed")

if __name__ == '__main__':
    app = wx.App()

    frm = HelloFrame(None, 'PRI_Project')

    frm.Show()
    app.MainLoop()