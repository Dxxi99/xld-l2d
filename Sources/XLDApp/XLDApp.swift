import SwiftUI
import AppKit
import UniformTypeIdentifiers

@main
struct XLDApp: App {
    @StateObject private var appState = AppState()

    var body: some Scene {
        WindowGroup("XLD-L2D") {
            ContentView()
                .environmentObject(appState)
                .preferredColorScheme(.dark)
                .frame(minWidth: 1400, minHeight: 900)
        }
        .windowStyle(.hiddenTitleBar)
        .commands {
            CommandGroup(replacing: .newItem) {
                Button("Open Image…") { openImageDialog() }
                    .keyboardShortcut("o", modifiers: .command)
                Divider()
                Button("Save Project") {}.keyboardShortcut("s", modifiers: .command).disabled(true)
                Button("Export…") {}.keyboardShortcut("e", modifiers: .command).disabled(true)
            }
            CommandGroup(before: .toolbar) {
                Button("Reset Pose") { appState.resetPose() }
                    .keyboardShortcut("0", modifiers: .command)
            }
            CommandMenu("Object") {
                Button("Add Layer") {}.disabled(true)
                Button("Add Behavior") {}.disabled(true)
                Button("Add Filter") {}.disabled(true)
            }
            CommandMenu("Markers") {
                Button("Add Marker") {}.disabled(true)
                Button("Delete Marker") {}.disabled(true)
            }
        }
    }

    private func openImageDialog() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.png, .jpeg, .tiff, .heic]
        panel.allowsMultipleSelection = false
        panel.canChooseDirectories = false
        if panel.runModal() == .OK, let url = panel.url {
            appState.openImage(url)
        }
    }
}
