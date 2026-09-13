import SwiftUI
import CoreGraphics
import XLDKit

@MainActor
final class TrackingDebugStore: ObservableObject {
    @Published var image: CGImage?
    @Published var skeleton: XLDTrackingSkeleton?
}

struct TrackingDebugView: View {
    @ObservedObject var store: TrackingDebugStore

    var body: some View {
        VStack(spacing: 0) {
            ZStack {
                Color.black
                if let img = store.image {
                    Image(decorative: img, scale: 1.0)
                        .resizable()
                        .interpolation(.none)
                        .aspectRatio(contentMode: .fit)
                } else {
                    Text("Waiting for camera...")
                        .foregroundColor(.gray)
                        .font(.title3)
                }
            }
            .frame(minWidth: 640, minHeight: 480)

            Divider()

            VStack(alignment: .leading, spacing: 4) {
                if let k = store.skeleton {
                    Text(String(format: "eyeL %.2f   eyeR %.2f", k.eyeLid_L, k.eyeLid_R))
                    Text(String(format: "irisL (%+.2f, %+.2f)  irisR (%+.2f, %+.2f)",
                                k.iris_L.x, k.iris_L.y, k.iris_R.x, k.iris_R.y))
                    Text(String(format: "browL %+.2f   browR %+.2f", k.brow_L, k.brow_R))
                    Text(String(format: "mouthO %.2f  mouthW %.2f  cornerL %+.2f  cornerR %+.2f",
                                k.mouthOpen, k.mouthWidth, k.mouthCorner_L, k.mouthCorner_R))
                } else {
                    Text("no skeleton data")
                }
            }
            .font(.system(.body, design: .monospaced))
            .padding(8)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }
}
