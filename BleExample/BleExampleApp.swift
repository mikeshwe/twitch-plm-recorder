//
//  TwitchRecorder — overnight PLM recorder for the WitMotion WT901BLECL
//
//  v2: dual-sensor support + Garmin HRM heart-rate strap integration.
//  Connect up to 2 WitMotion sensors (ankle/chest) and a standard
//  Bluetooth heart-rate monitor (Garmin HRM-Dual etc.). CSV records
//  ankle, chest, and HR data on the same timer tick.
//

import SwiftUI
import CoreBluetooth
import WitSDK
import AVFoundation


// **********************************************************
// MARK: HRMManager — talks to a standard BLE Heart Rate Service
// **********************************************************
/// The Bluetooth SIG defines the Heart Rate Service (0x180D) with the
/// Heart Rate Measurement characteristic (0x2A37). Garmin HRM-Dual,
/// HRM-Pro, Polar straps, and most others expose it the same way.
/// This class owns its own CBCentralManager because the WitMotion SDK's
/// manager isn't a good fit for standard GATT services.
class HRMManager: NSObject, ObservableObject, CBCentralManagerDelegate, CBPeripheralDelegate {

    static let heartRateServiceUUID = CBUUID(string: "180D")
    static let heartRateMeasurementCharacteristicUUID = CBUUID(string: "2A37")
    static let batteryServiceUUID = CBUUID(string: "180F")
    static let batteryLevelCharacteristicUUID = CBUUID(string: "2A19")

    @Published var enableScan = false
    @Published var discovered: [CBPeripheral] = []
    @Published var connected: CBPeripheral?
    @Published var lastHR: Int?
    @Published var lastRR: Int?           // most recent RR-interval, ms
    @Published var lastBattery: Int?
    @Published var liveSnapshot: String = "Not connected"
    /// Walltime when the last HR notification arrived. Used to detect
    /// stale readings (strap off-skin or BLE timeout).
    @Published var lastHRReceivedAt: Date?
    /// Walltime of last notification that actually carried RR data.
    /// When this lags lastHRReceivedAt by > a few seconds, the strap is
    /// reporting HR but not detecting individual beats reliably — usually
    /// dry/loose electrodes.
    @Published var lastRRReceivedAt: Date?
    /// True when no HR notification has arrived in STALE_AFTER_S seconds.
    @Published var hrIsStale: Bool = false

    /// True when HR is updating but RR data is older than RR_STALE_AFTER_S.
    /// Means: strap is broadcasting an HR estimate but isn't detecting
    /// individual beats — diagnostic for poor electrode contact.
    var rrIsStale: Bool {
        guard let rrAt = lastRRReceivedAt else { return lastHRReceivedAt != nil }
        return Date().timeIntervalSince(rrAt) > HRMManager.RR_STALE_AFTER_S
    }

    static let RR_STALE_AFTER_S: TimeInterval = 5.0

    /// HR data is "stale" if no notification has arrived in this long.
    /// Per Bluetooth HR Service spec the strap sends ~1 per heartbeat;
    /// 5s without a beat means HR < 12 bpm (impossible) so almost
    /// certainly the strap is off-skin or disconnected.
    static let STALE_AFTER_S: TimeInterval = 5.0

    private var central: CBCentralManager!
    private var hrmCharacteristic: CBCharacteristic?
    private var staleTimer: Timer?

    override init() {
        super.init()
        // No state restoration: iOS rejects the restore identifier without
        // additional entitlements that would require deeper signing setup.
        // The `bluetooth-central` background mode + DispatchSourceTimer
        // gives us enough background resilience for overnight recordings;
        // we just won't survive a full app kill the way a state-restoring
        // app would.
        central = CBCentralManager(delegate: self, queue: .main)
    }

    // MARK: -- Scan / connect ---------------------------------------------

    func scan() {
        guard central.state == .poweredOn else { return }
        discovered.removeAll()
        // Filter by HR service so we don't see every BLE device on the phone
        central.scanForPeripherals(withServices: [HRMManager.heartRateServiceUUID])
        enableScan = true
    }

    func stopScan() {
        central.stopScan()
        enableScan = false
    }

    func connect(_ peripheral: CBPeripheral) {
        stopScan()
        peripheral.delegate = self
        central.connect(peripheral)
    }

    func disconnect() {
        if let p = connected { central.cancelPeripheralConnection(p) }
        connected = nil
        hrmCharacteristic = nil
        lastHR = nil
        lastRR = nil
        lastBattery = nil
        lastHRReceivedAt = nil
        lastRRReceivedAt = nil
        hrIsStale = false
        staleTimer?.invalidate()
        staleTimer = nil
        liveSnapshot = "Not connected"
    }

    // MARK: -- CBCentralManagerDelegate -----------------------------------

    func centralManagerDidUpdateState(_ central: CBCentralManager) {
        // ready to scan once .poweredOn
    }

    func centralManager(_ central: CBCentralManager,
                        didDiscover peripheral: CBPeripheral,
                        advertisementData: [String : Any], rssi RSSI: NSNumber) {
        if !discovered.contains(where: { $0.identifier == peripheral.identifier }) {
            discovered.append(peripheral)
        }
    }

    func centralManager(_ central: CBCentralManager, didConnect peripheral: CBPeripheral) {
        connected = peripheral
        peripheral.discoverServices([HRMManager.heartRateServiceUUID,
                                     HRMManager.batteryServiceUUID])
        liveSnapshot = "Connected, discovering services…"
    }

    func centralManager(_ central: CBCentralManager, didDisconnectPeripheral peripheral: CBPeripheral, error: Error?) {
        // Auto-reconnect on unexpected disconnect — the strap may briefly
        // lose contact and reestablish, especially during rolling. We don't
        // clear `connected` until the user explicitly disconnects or the
        // peripheral is unrecoverable.
        liveSnapshot = "Disconnected — reconnecting…"
        hrmCharacteristic = nil
        lastHRReceivedAt = nil
        hrIsStale = false
        staleTimer?.invalidate()
        staleTimer = nil
        // Try to reconnect for up to 30 minutes
        if peripheral == connected {
            central.connect(peripheral)
        }
    }

    // MARK: -- CBPeripheralDelegate ---------------------------------------

    func peripheral(_ peripheral: CBPeripheral, didDiscoverServices error: Error?) {
        for service in peripheral.services ?? [] {
            if service.uuid == HRMManager.heartRateServiceUUID {
                peripheral.discoverCharacteristics(
                    [HRMManager.heartRateMeasurementCharacteristicUUID], for: service)
            } else if service.uuid == HRMManager.batteryServiceUUID {
                peripheral.discoverCharacteristics(
                    [HRMManager.batteryLevelCharacteristicUUID], for: service)
            }
        }
    }

    func peripheral(_ peripheral: CBPeripheral,
                    didDiscoverCharacteristicsFor service: CBService,
                    error: Error?) {
        for ch in service.characteristics ?? [] {
            if ch.uuid == HRMManager.heartRateMeasurementCharacteristicUUID {
                hrmCharacteristic = ch
                peripheral.setNotifyValue(true, for: ch)
            } else if ch.uuid == HRMManager.batteryLevelCharacteristicUUID {
                peripheral.readValue(for: ch)
                peripheral.setNotifyValue(true, for: ch)
            }
        }
    }

    func peripheral(_ peripheral: CBPeripheral,
                    didUpdateValueFor characteristic: CBCharacteristic,
                    error: Error?) {
        guard let data = characteristic.value else { return }
        if characteristic.uuid == HRMManager.heartRateMeasurementCharacteristicUUID {
            parseHeartRate(data)
        } else if characteristic.uuid == HRMManager.batteryLevelCharacteristicUUID,
                  let firstByte = data.first {
            DispatchQueue.main.async { self.lastBattery = Int(firstByte) }
        }
    }

    /// Parse the Heart Rate Measurement characteristic per Bluetooth SIG spec.
    /// Layout: [flags][HR][energyExpended?][RR-intervals...]
    private func parseHeartRate(_ data: Data) {
        guard data.count >= 2 else { return }
        let flags = data[0]
        let hr16Bit = (flags & 0x01) != 0
        let energyPresent = (flags & 0x08) != 0
        let rrPresent = (flags & 0x10) != 0

        var offset = 1
        let hr: Int
        if hr16Bit {
            guard data.count >= offset + 2 else { return }
            hr = Int(data[offset]) | (Int(data[offset + 1]) << 8)
            offset += 2
        } else {
            hr = Int(data[offset])
            offset += 1
        }
        if energyPresent { offset += 2 }   // skip 16-bit energy expended

        // Collect any RR-intervals (1/1024 second units). Filter physiologically
        // implausible values: real human RR is 300-2000 ms (HR 30-200 bpm).
        // Values outside that range come from sensor noise (typically a dying
        // strap battery causing beat-detection misfires).
        var rrIntervals: [Int] = []
        if rrPresent {
            while offset + 1 < data.count {
                let raw = Int(data[offset]) | (Int(data[offset + 1]) << 8)
                let ms = Int(round(Double(raw) * 1000.0 / 1024.0))
                if ms >= 300 && ms <= 2000 {
                    rrIntervals.append(ms)
                }
                offset += 2
            }
        }
        // If HR is way out of physiological range, discard the whole notification
        if hr < 30 || hr > 220 {
            return
        }

        DispatchQueue.main.async {
            self.lastHR = hr
            // Only update lastRR when this notification actually included
            // an RR value. If absent, keep the previous good value — the
            // strap omits RR when it's not confident in beat timing, but
            // we still want to show the last detected interval.
            if let newRR = rrIntervals.last {
                self.lastRR = newRR
                self.lastRRReceivedAt = Date()
            }
            self.lastHRReceivedAt = Date()
            self.hrIsStale = false
            let rrText = self.lastRR.map { "\($0) ms" } ?? "—"
            self.liveSnapshot = "HR: \(hr) bpm\nRR: \(rrText)"
            self.startStaleWatcher()
        }
    }

    /// Run a 1 Hz timer that flips `hrIsStale = true` if no HR notifications
    /// have arrived for STALE_AFTER_S, and forces a re-render so `rrIsStale`
    /// (a computed property) updates in the UI when RR goes stale even if
    /// HR keeps arriving.
    private func startStaleWatcher() {
        staleTimer?.invalidate()
        staleTimer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { [weak self] _ in
            guard let self = self else { return }
            // HR staleness check
            if let last = self.lastHRReceivedAt {
                let stale = Date().timeIntervalSince(last) > HRMManager.STALE_AFTER_S
                if stale != self.hrIsStale {
                    DispatchQueue.main.async { self.hrIsStale = stale }
                }
            }
            // Force a tick on every second so rrIsStale (computed) is
            // re-evaluated by the UI. We do this by toggling a hidden
            // dummy @Published when RR-staleness state changes.
            let rrStaleNow = self.rrIsStale
            if rrStaleNow != self.lastRRStaleSnapshot {
                self.lastRRStaleSnapshot = rrStaleNow
                DispatchQueue.main.async { self.objectWillChange.send() }
            }
        }
    }

    private var lastRRStaleSnapshot: Bool = false
}


// **********************************************************
// MARK: Calibration state machine
// **********************************************************
/// Drives the in-app calibration wizard.
///
/// Flow:
///   idle
///     → countdown(pos, 3…1)   "Get ready for BACK"
///     → holding(pos, 8…1)     "Hold BACK — 7s"  (recording running)
///     → countdown(next, 3…1)  "Get ready for RIGHT"
///     → holding(next, 8…1)
///     … repeat for all 3 positions …
///     → done(csvURL)           auto-saved, ready to share
enum CalibrationPhase: Equatable {
    case idle
    case countdown(position: String, secondsLeft: Int)   // "get ready" pause
    case holding(position: String, secondsLeft: Int)     // recording this position
    // No .done state — wizard transitions directly back to .idle, and the
    // night recording continues in the same file without interruption.

    var isActive: Bool {
        switch self { case .idle: return false; default: return true }
    }
}

// **********************************************************
// MARK: Sensor roles
// **********************************************************
enum SensorRole: String, CaseIterable, Identifiable {
    case rightAnkle, leftAnkle
    var id: String { rawValue }
    var label: String {
        switch self {
        case .rightAnkle: return "Right ankle"
        case .leftAnkle: return "Left ankle"
        }
    }
    /// Short prefix used in CSV column names: "ankleR", "ankleL".
    var csvPrefix: String {
        switch self {
        case .rightAnkle: return "ankleR"
        case .leftAnkle: return "ankleL"
        }
    }
}


// **********************************************************
// MARK: BackgroundAudioKeeper — keeps the app alive overnight
// **********************************************************
//
// Why this exists: iOS suspends apps once the screen turns off, even with
// `bluetooth-central` background mode declared. The only reliable way to
// keep our recording timer firing through a full night is to look like a
// "media playback" app — iOS gives those special background privileges.
//
// We do this by playing a silent audio loop continuously while recording.
// The audio session is configured with `mixWithOthers` so it doesn't
// interrupt or duck other audio (specifically, the user's Calm app
// streaming sleep sounds via Bluetooth). To iOS we look like an audio
// player; to the user nothing audible is happening.
class BackgroundAudioKeeper: ObservableObject {

    @Published var isActive: Bool = false
    @Published var lastError: String?

    private var player: AVAudioPlayer?
    private var observer: NSObjectProtocol?

    /// Configure the audio session and start an indefinite near-silent loop.
    /// Idempotent — calling start() while already running has no effect.
    ///
    /// iOS 17+ detects truly silent streams (zero volume + zero-energy PCM)
    /// and can revoke background privileges. We use a 1 Hz sine wave at
    /// 0.1% volume (-60 dB) — inaudible in practice but unambiguously
    /// "real" audio to the OS. Combined with .mixWithOthers this doesn't
    /// interfere with Calm or any other sleep audio.
    func start() {
        guard player == nil else { return }
        do {
            let session = AVAudioSession.sharedInstance()
            // Playback category = "real" media app, gets background privileges.
            // mixWithOthers = don't interrupt Calm's audio streaming to the
            // sleep mask. duckOthers absent = our silent stream doesn't
            // lower their volume either.
            try session.setCategory(.playback, mode: .default,
                                    options: [.mixWithOthers])
            try session.setActive(true, options: [])

            // 1 Hz sine at -60 dB: inaudible but undeniably real audio to iOS.
            let toneWAV = makeToneWAV(durationSeconds: 1.0, frequencyHz: 1.0, amplitude: 0.001)
            let p = try AVAudioPlayer(data: toneWAV)
            p.numberOfLoops = -1   // loop forever
            p.volume = 1.0         // volume on player; amplitude in PCM is already tiny
            p.prepareToPlay()
            p.play()
            player = p

            // Re-establish playback after interruptions (incoming call,
            // alarm, etc.). iOS pauses us on interruption begin; we
            // resume on interruption end.
            observer = NotificationCenter.default.addObserver(
                forName: AVAudioSession.interruptionNotification,
                object: session,
                queue: .main
            ) { [weak self] notification in
                self?.handleInterruption(notification)
            }

            isActive = true
            lastError = nil
            print("BackgroundAudioKeeper: silent loop started")
        } catch {
            isActive = false
            lastError = error.localizedDescription
            print("BackgroundAudioKeeper: failed to start: \(error)")
        }
    }

    func stop() {
        player?.stop()
        player = nil
        if let obs = observer {
            NotificationCenter.default.removeObserver(obs)
            observer = nil
        }
        try? AVAudioSession.sharedInstance().setActive(false, options: [.notifyOthersOnDeactivation])
        isActive = false
        print("BackgroundAudioKeeper: stopped")
    }

    private func handleInterruption(_ notification: Notification) {
        guard let info = notification.userInfo,
              let typeValue = info[AVAudioSessionInterruptionTypeKey] as? UInt,
              let type = AVAudioSession.InterruptionType(rawValue: typeValue)
        else { return }

        switch type {
        case .began:
            isActive = false
            print("BackgroundAudioKeeper: interruption began (another app took audio session)")
            // Schedule a reclaim attempt after a short delay. Many apps
            // (Eight Sleep, etc.) grab the session briefly then release it.
            // We don't wait for .ended because some apps never send it.
            DispatchQueue.main.asyncAfter(deadline: .now() + 2.0) { [weak self] in
                self?.reclaimAudioSession()
            }
        case .ended:
            // Always try to reclaim regardless of shouldResume flag —
            // some apps end the interruption without setting that flag.
            print("BackgroundAudioKeeper: interruption ended — reclaiming session")
            reclaimAudioSession()
        @unknown default:
            break
        }
    }

    /// Forcibly reclaim the audio session and restart playback.
    /// Called after interruptions and periodically by the app watchdog.
    func reclaimAudioSession() {
        guard player != nil else { return }   // not started, nothing to reclaim
        do {
            try AVAudioSession.sharedInstance().setActive(true, options: [])
            if player?.isPlaying == false {
                player?.play()
            }
            isActive = true
            print("BackgroundAudioKeeper: audio session reclaimed")
        } catch {
            isActive = false
            lastError = "reclaim failed: \(error.localizedDescription)"
            print("BackgroundAudioKeeper: reclaim failed: \(error)")
        }
    }

    /// Build a 16-bit mono PCM WAV containing a sine tone.
    /// Default: 1 Hz at amplitude 0.001 (-60 dBFS) — inaudible but real
    /// audio energy that iOS 17+ won't classify as a "silent stream" and
    /// revoke background playback privileges for.
    private func makeToneWAV(durationSeconds: Double,
                              frequencyHz: Double = 1.0,
                              amplitude: Double = 0.001) -> Data {
        let sampleRate: UInt32 = 8000
        let channels: UInt16 = 1
        let bitsPerSample: UInt16 = 16
        let numSamples = Int(Double(sampleRate) * durationSeconds)
        let dataSize = UInt32(numSamples * 2)   // 2 bytes per 16-bit sample
        let chunkSize = 36 + dataSize

        var d = Data()
        // RIFF header
        d.append(contentsOf: [0x52, 0x49, 0x46, 0x46])  // "RIFF"
        d.append(uint32LE(chunkSize))
        d.append(contentsOf: [0x57, 0x41, 0x56, 0x45])  // "WAVE"
        // fmt chunk
        d.append(contentsOf: [0x66, 0x6d, 0x74, 0x20])  // "fmt "
        d.append(uint32LE(16))                            // fmt chunk size
        d.append(uint16LE(1))                             // PCM
        d.append(uint16LE(channels))
        d.append(uint32LE(sampleRate))
        d.append(uint32LE(sampleRate * 2))                // byte rate
        d.append(uint16LE(2))                             // block align
        d.append(uint16LE(bitsPerSample))
        // data chunk
        d.append(contentsOf: [0x64, 0x61, 0x74, 0x61])  // "data"
        d.append(uint32LE(dataSize))
        // 16-bit signed PCM sine samples
        let peak = amplitude * Double(Int16.max)
        for i in 0 ..< numSamples {
            let phase = 2.0 * Double.pi * frequencyHz * Double(i) / Double(sampleRate)
            let sample = Int16(clamping: Int(peak * sin(phase)))
            d.append(uint16LE(UInt16(bitPattern: sample)))
        }
        return d
    }

    private func uint16LE(_ v: UInt16) -> Data {
        Data([UInt8(v & 0xff), UInt8(v >> 8 & 0xff)])
    }
    private func uint32LE(_ v: UInt32) -> Data {
        Data([UInt8(v & 0xff), UInt8(v >> 8 & 0xff),
              UInt8(v >> 16 & 0xff), UInt8(v >> 24 & 0xff)])
    }
}


/// Wraps a connected device with its assigned role and a live snapshot
/// for the UI. The actual SDK device is held by reference; the rest of
/// the struct is observable state.
class SensorSlot: ObservableObject, Identifiable {
    let id = UUID()
    @Published var device: Bwt901ble?
    @Published var liveSnapshot: String = "Not connected"
    @Published var lastBattery: String = "—"
    let role: SensorRole

    init(role: SensorRole) {
        self.role = role
    }

    var isConnected: Bool { device != nil }
}


// **********************************************************
// MARK: App entry point
// **********************************************************
@main
struct TwitchRecorderApp: App {

    @StateObject var ctx = AppContext()

    var body: some Scene {
        WindowGroup {
            NavigationView {
                MainView()
                    .environmentObject(ctx)
            }
        }
    }
}


// **********************************************************
// MARK: AppContext — owns BLE, slots, and recording state
// **********************************************************
class AppContext: ObservableObject, IBluetoothEventObserver, IBwt901bleRecordObserver {

    // BLE
    var bluetoothManager: WitBluetoothManager = WitBluetoothManager.instance
    @Published var enableScan = false
    @Published var deviceList: [Bwt901ble] = []

    // Two slots — ankle (PLM detection) and chest (position + breathing)
    @Published var rightAnkle = SensorSlot(role: .rightAnkle)
    @Published var leftAnkle = SensorSlot(role: .leftAnkle)

    // Heart-rate strap (Garmin HRM-Dual etc.) — separate manager because
    // it speaks standard BLE GATT, not WitMotion's protocol.
    @Published var hrm = HRMManager()

    // Silent audio loop to keep the app alive in background. See class
    // comment for details on the iOS audio-session trick.
    @Published var audioKeeper = BackgroundAudioKeeper()

    /// When the user taps a discovered device, we ask which role to assign.
    /// Holds the device awaiting role selection.
    @Published var pendingDevice: Bwt901ble?

    // Recording
    @Published var isRecording = false
    @Published var sampleCount: Int = 0
    @Published var sessionStartedAt: Date?
    @Published var lastExportURL: URL?

    // Calibration wizard (runs as opening phase of each overnight recording)
    @Published var calibrationPhase: CalibrationPhase = .idle

    /// 30 Hz polling rate — see comment from v0. BLE bandwidth caps real
    /// throughput at ~24 Hz per sensor, so 30 Hz polling captures every
    /// new sample with minimal duplication. With 2 sensors on a phone's
    /// CoreBluetooth stack we still want to keep this rate (the radio
    /// can multiplex two connections without halving throughput).
    static let recordingHz: Double = 30

    private var recordingTimer: DispatchSourceTimer?
    private var csvHandle: FileHandle?
    /// Background-task identifier from beginBackgroundTask. Kept renewed
    /// while recording so iOS treats us as actively-needed and doesn't
    /// suspend the recording queue. Combined with the audio session, this
    /// is belt-and-suspenders for staying alive overnight.
    private var bgTaskID: UIBackgroundTaskIdentifier = .invalid
    /// Independent watchdog timer running on the main run loop. Its job
    /// is to renew the background task every 25 seconds and to detect if
    /// the recording timer has stopped firing (which can happen under
    /// iOS background CPU throttling). Independent of the recording
    /// queue so a hang there doesn't take this down.
    private var watchdogTimer: Timer?
    private var lastWatchdogSampleCount: Int = 0
    private var watchdogStallTicks: Int = 0
    /// Single recording queue kept alive for the app's lifetime. Reused by
    /// the watchdog's restartRecordingTimer() so we never leak queues — a
    /// leaked DispatchQueue holds a thread-pool thread and a kernel queue
    /// object, which accumulates into an OOM kill after several hours.
    private let recordingQueue = DispatchQueue(label: "twitch.recording", qos: .userInitiated)
    private var currentSessionURL: URL?
    private let dateFormatter: ISO8601DateFormatter = {
        let f = ISO8601DateFormatter()
        f.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return f
    }()

    init() {
        startRefreshThread()
        // After a crash or force-quit during the previous stopRecording(),
        // surface the most recent CSV so the user can still share it.
        recoverLastSession()
    }

    // MARK: -- Slot lookup helpers ----------------------------------------

    func slot(for role: SensorRole) -> SensorSlot {
        role == .rightAnkle ? rightAnkle : leftAnkle
    }

    var connectedSlots: [SensorSlot] {
        [rightAnkle, leftAnkle].filter { $0.isConnected }
    }

    var hasAnyDataSource: Bool {
        !connectedSlots.isEmpty || hrm.connected != nil
    }

    // MARK: -- Scanning / connection --------------------------------------

    func scanDevices() {
        // Clear the discoverable list; keep already-connected slots intact
        deviceList.removeAll()
        bluetoothManager.registerEventObserver(observer: self)
        bluetoothManager.startScan()
        enableScan = true
    }

    func stopScan() {
        bluetoothManager.removeEventObserver(observer: self)
        bluetoothManager.stopScan()
        enableScan = false
    }

    func onFoundBle(bluetoothBLE: BluetoothBLE?) {
        guard isNotKnown(bluetoothBLE) else { return }
        DispatchQueue.main.async {
            self.deviceList.append(Bwt901ble(bluetoothBLE: bluetoothBLE))
        }
    }

    /// Skip devices we're already tracking — either in the discovered list,
    /// or already assigned to a slot.
    private func isNotKnown(_ ble: BluetoothBLE?) -> Bool {
        guard let mac = ble?.mac else { return false }
        for d in deviceList where d.mac == mac { return false }
        if rightAnkle.device?.mac == mac { return false }
        if leftAnkle.device?.mac == mac { return false }
        return true
    }

    func onConnected(bluetoothBLE: BluetoothBLE?) {
        print("BLE connected: \(bluetoothBLE?.peripheral.name ?? "?")")
    }

    func onConnectionFailed(bluetoothBLE: BluetoothBLE?) {
        print("BLE connection failed: \(bluetoothBLE?.peripheral.name ?? "?")")
    }

    func onDisconnected(bluetoothBLE: BluetoothBLE?) {
        print("BLE disconnected: \(bluetoothBLE?.peripheral.name ?? "?")")
        // Find which slot this was, mark it as disconnected, schedule retry
        DispatchQueue.main.async {
            var droppedSlot: SensorSlot?
            if self.rightAnkle.device?.mac == bluetoothBLE?.mac {
                droppedSlot = self.rightAnkle
            } else if self.leftAnkle.device?.mac == bluetoothBLE?.mac {
                droppedSlot = self.leftAnkle
            }
            if let slot = droppedSlot, let device = slot.device {
                slot.liveSnapshot = "Disconnected — reconnecting…"
                // Attempt auto-reconnect during recording. If we're not
                // recording, just clear the slot (user disconnected manually
                // or sensor was switched off).
                if self.isRecording {
                    self.scheduleReconnect(for: slot, device: device)
                } else {
                    slot.device = nil
                    slot.liveSnapshot = "Disconnected"
                }
            }
            if self.isRecording && !self.hasAnyDataSource {
                self.stopRecording()
            }
        }
    }

    /// Track in-flight reconnect attempts so we don't pile up duplicates
    private var reconnectingMacs: Set<String> = []

    /// Try to reopen a disconnected sensor. Retries every 30s for up to 30
    /// minutes (60 attempts) before giving up. Most BLE disconnects are
    /// transient (sensor moved out of range, brief radio interference) and
    /// recover within seconds.
    private func scheduleReconnect(for slot: SensorSlot, device: Bwt901ble, attempt: Int = 0) {
        guard let mac = device.mac else { return }
        if attempt == 0 {
            guard !reconnectingMacs.contains(mac) else { return }
            reconnectingMacs.insert(mac)
        }
        guard attempt < 60 else {
            print("Giving up reconnect for \(device.name ?? "?") after \(attempt) attempts")
            DispatchQueue.main.async {
                slot.device = nil
                slot.liveSnapshot = "Disconnected (gave up)"
                self.reconnectingMacs.remove(mac)
            }
            return
        }
        DispatchQueue.main.asyncAfter(deadline: .now() + 30.0) { [weak self] in
            guard let self = self else { return }
            // If recording stopped or user manually reassigned, abort
            guard self.isRecording, slot.device?.mac == mac || slot.device == nil else {
                self.reconnectingMacs.remove(mac)
                return
            }
            do {
                try device.openDevice()
                device.registerListenKeyUpdateObserver(obj: self)
                slot.device = device
                slot.liveSnapshot = "Reconnected (attempt \(attempt + 1))"
                print("Reconnected \(device.name ?? "?") on attempt \(attempt + 1)")
                self.reconnectingMacs.remove(mac)
            } catch {
                print("Reconnect attempt \(attempt + 1) failed for \(device.name ?? "?"): \(error)")
                self.scheduleReconnect(for: slot, device: device, attempt: attempt + 1)
            }
        }
    }

    /// User tapped a discovered device — present the role-picker first.
    func requestAssignment(_ device: Bwt901ble) {
        pendingDevice = device
    }

    /// User picked a role for the pending device. Open it and assign to slot.
    func assignPending(to role: SensorRole) {
        guard let device = pendingDevice else { return }
        pendingDevice = nil

        // Don't allow overwriting an already-connected slot — disconnect
        // first if needed.
        let target = slot(for: role)
        if let existing = target.device {
            existing.closeDevice()
            target.device = nil
        }

        do {
            try device.openDevice()
            device.registerListenKeyUpdateObserver(obj: self)
            DispatchQueue.main.async {
                target.device = device
                print("Assigned \(device.name ?? "?") to \(role.rawValue)")
            }
            // Remove from the discoverable list since it's now claimed
            deviceList.removeAll { $0.mac == device.mac }
            // Stop scanning if both slots are now full
            if connectedSlots.count >= 2 { stopScan() }
        } catch {
            print("openDevice failed: \(error)")
        }
    }

    func cancelAssignment() {
        pendingDevice = nil
    }

    func disconnect(_ slot: SensorSlot) {
        if let d = slot.device { d.closeDevice() }
        slot.device = nil
        slot.liveSnapshot = "Not connected"
    }

    /// Set output rate on a single device (or all connected if device is nil).
    /// 0x06=10Hz, 0x07=20Hz, 0x08=50Hz, 0x09=100Hz, 0x0B=200Hz.
    func setSensorOutputRate(_ rateCode: UInt8, for device: Bwt901ble? = nil) {
        let targets: [Bwt901ble] = device.map { [$0] } ?? connectedSlots.compactMap(\.device)
        for d in targets {
            DispatchQueue.global(qos: .userInitiated).async {
                do {
                    try d.unlockReg()
                    try d.writeRge([0xFF, 0xAA, 0x03, rateCode, 0x00], 100)
                    try d.saveReg()
                    print("Output rate 0x\(String(rateCode, radix: 16)) set on \(d.name ?? "?")")
                } catch {
                    print("setSensorOutputRate failed for \(d.name ?? "?"): \(error)")
                }
            }
        }
    }

    // MARK: -- Calibration wizard -----------------------------------------
    //
    // Calibration is the FIRST phase of a normal overnight recording.
    // Tapping "Start recording" opens the night's CSV and immediately
    // runs the 3-position wizard (~33s). When the last position finishes,
    // the wizard sets calibrationPhase = .idle and recording continues
    // seamlessly into the night — same file, same timer, no interruption.
    //
    // The result is one CSV per night that contains:
    //   • rows 1–~1000  : calibration still-windows (back / right / left)
    //   • rows 1001–end : overnight PLM data
    //
    // calibrate_position.py picks up the still windows from the first
    // ~33 seconds; analyze_night.py uses the rest.

    static let calibrationPositions       = ["back", "right", "left"]
    static let calibrationHoldSeconds     = 10  // seconds still per position
    static let calibrationCountdownSeconds = 10  // "get ready" pause between positions

    private var calibrationTimer: Timer?
    private var calibrationPositionIndex = 0
    private var calibrationTicksLeft     = 0

    /// Called by startRecording() after the file and timer are running.
    /// Drives the UI state machine; the recording timer keeps writing rows
    /// throughout — caller doesn't need to do anything else.
    private func startCalibrationWizard() {
        startCalibrationCountdown(positionIndex: 0)
    }

    private func startCalibrationCountdown(positionIndex: Int) {
        calibrationPositionIndex = positionIndex
        let pos = Self.calibrationPositions[positionIndex]
        calibrationTicksLeft = Self.calibrationCountdownSeconds
        calibrationPhase = .countdown(position: pos, secondsLeft: calibrationTicksLeft)
        UIImpactFeedbackGenerator(style: .medium).impactOccurred()

        calibrationTimer?.invalidate()
        calibrationTimer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { [weak self] _ in
            guard let self = self else { return }
            self.calibrationTicksLeft -= 1
            if self.calibrationTicksLeft > 0 {
                let p = Self.calibrationPositions[self.calibrationPositionIndex]
                self.calibrationPhase = .countdown(position: p, secondsLeft: self.calibrationTicksLeft)
            } else {
                self.calibrationTimer?.invalidate()
                self.startCalibrationHold(positionIndex: self.calibrationPositionIndex)
            }
        }
        RunLoop.main.add(calibrationTimer!, forMode: .common)
    }

    private func startCalibrationHold(positionIndex: Int) {
        let pos = Self.calibrationPositions[positionIndex]
        calibrationTicksLeft = Self.calibrationHoldSeconds
        calibrationPhase = .holding(position: pos, secondsLeft: calibrationTicksLeft)
        UIImpactFeedbackGenerator(style: .heavy).impactOccurred()

        calibrationTimer?.invalidate()
        calibrationTimer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { [weak self] _ in
            guard let self = self else { return }
            self.calibrationTicksLeft -= 1
            if self.calibrationTicksLeft > 0 {
                let p = Self.calibrationPositions[self.calibrationPositionIndex]
                self.calibrationPhase = .holding(position: p, secondsLeft: self.calibrationTicksLeft)
            } else {
                self.calibrationTimer?.invalidate()
                UINotificationFeedbackGenerator().notificationOccurred(.success)
                let nextIndex = positionIndex + 1
                if nextIndex < Self.calibrationPositions.count {
                    self.startCalibrationCountdown(positionIndex: nextIndex)
                } else {
                    // All positions done — transition to overnight recording.
                    // The recording timer is already running; just clear the
                    // wizard state so the UI shows the normal recording panel.
                    self.calibrationPhase = .idle
                    print("Calibration complete — continuing overnight recording")
                }
            }
        }
        RunLoop.main.add(calibrationTimer!, forMode: .common)
    }

    func cancelCalibration() {
        // Cancelling during calibration = cancel the whole recording session,
        // since calibration is the opening phase of the night file.
        calibrationTimer?.invalidate()
        calibrationTimer = nil
        calibrationPhase = .idle
        stopRecording()
    }

    // MARK: -- Recording --------------------------------------------------

    func onRecord(_ bwt901ble: Bwt901ble) {}

    /// Read all sources at this tick and write one combined CSV row.
    /// Missing source columns are written as empty strings — the analysis
    /// script handles partial rows naturally.
    private func sampleAndWrite() {
        guard isRecording, let handle = csvHandle else { return }

        let ts = dateFormatter.string(from: Date())
        let rRow = sensorRowFields(for: rightAnkle.device)
        let lRow = sensorRowFields(for: leftAnkle.device)
        // HR columns retained in CSV for backward-compat; left blank when
        // no HRM is connected (current dual-ankle setup, no chest strap).
        let hr = (hrm.hrIsStale ? "" : hrm.lastHR.map(String.init) ?? "")
        let rr = (hrm.hrIsStale ? "" : hrm.lastRR.map(String.init) ?? "")

        let row = "\(ts),\(rRow),\(lRow),\(hr),\(rr)\n"
        if let data = row.data(using: .utf8) {
            handle.write(data)
        }
        // Track count on the recording queue, push to main once per second
        // for the UI counter (sampleCount is @Published, must be main-thread).
        backgroundSampleCount += 1
        if backgroundSampleCount - lastUISampleCount >= 30 {
            let snapshot = backgroundSampleCount
            lastUISampleCount = snapshot
            DispatchQueue.main.async { [weak self] in
                self?.sampleCount = snapshot
            }
        }
        // Renew background task every ~30s. iOS doesn't always honor the
        // expirationHandler if it decides we're misbehaving, so we
        // defensively re-up our background time from the recording loop.
        if backgroundSampleCount % 900 == 0 {
            DispatchQueue.main.async { [weak self] in
                self?.beginBackgroundTask()
            }
        }
    }

    /// Sample counter on the recording queue (no main-thread synchronization).
    /// We push it to `sampleCount` (@Published, main-thread only) once per
    /// second to update the live UI counter without burning main-thread cycles.
    private var backgroundSampleCount: Int = 0
    private var lastUISampleCount: Int = 0

    /// 6 comma-separated fields in column order: ax,ay,az,gx,gy,gz.
    /// Empty string fields when the device is nil.
    private func sensorRowFields(for device: Bwt901ble?) -> String {
        guard let d = device else { return ",,,,," }
        let ax = d.getDeviceData(WitSensorKey.AccX) ?? ""
        let ay = d.getDeviceData(WitSensorKey.AccY) ?? ""
        let az = d.getDeviceData(WitSensorKey.AccZ) ?? ""
        let gx = d.getDeviceData(WitSensorKey.GyroX) ?? ""
        let gy = d.getDeviceData(WitSensorKey.GyroY) ?? ""
        let gz = d.getDeviceData(WitSensorKey.GyroZ) ?? ""
        return "\(ax),\(ay),\(az),\(gx),\(gy),\(gz)"
    }

    func startRecording() {
        // Record if we have any data source (sensor or HRM)
        guard !isRecording, hasAnyDataSource else { return }

        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        let stamp = filenameStamp(Date())
        let url = docs.appendingPathComponent("twitch_\(stamp).csv")

        // 14-column CSV (timestamp + 6 right ankle + 6 left ankle + hr + rr)
        let header = "timestamp_iso," +
            "ankleR_acc_x_g,ankleR_acc_y_g,ankleR_acc_z_g,ankleR_gyro_x_dps,ankleR_gyro_y_dps,ankleR_gyro_z_dps," +
            "ankleL_acc_x_g,ankleL_acc_y_g,ankleL_acc_z_g,ankleL_gyro_x_dps,ankleL_gyro_y_dps,ankleL_gyro_z_dps," +
            "hr_bpm,rr_interval_ms\n"
        FileManager.default.createFile(atPath: url.path, contents: header.data(using: .utf8))

        do {
            csvHandle = try FileHandle(forWritingTo: url)
            try csvHandle?.seekToEnd()
        } catch {
            print("Failed to open CSV for writing: \(error)")
            return
        }

        currentSessionURL = url
        sampleCount = 0
        backgroundSampleCount = 0
        lastUISampleCount = 0
        sessionStartedAt = Date()
        isRecording = true
        print("Recording \(connectedSlots.count) sensor(s) to: \(url.path) at \(Self.recordingHz) Hz")

        // Start silent audio loop so iOS treats us as a background-audio
        // app and keeps the recording timer firing all night.
        audioKeeper.start()

        // Belt-and-suspenders: also begin a background task. iOS gives us
        // up to ~30s of guaranteed background time per begin call. We
        // renew it periodically (in sampleAndWrite()) so it never expires.
        beginBackgroundTask()

        // Start independent watchdog: renews the bgtask and checks that
        // the recording timer is still firing. If the recording queue
        // stalls under iOS throttling, the watchdog detects it and
        // restarts the recording timer.
        startWatchdog()

        // DispatchSourceTimer on a dedicated background queue. Unlike
        // NSTimer (which lives on the main run loop and pauses when iOS
        // suspends the app), this stays active as long as the process
        // is alive — including during backgrounded BLE-central operation.
        // We reuse recordingQueue (created once at init) so watchdog
        // restarts don't leak a new queue object each time.
        let interval = 1.0 / Self.recordingHz
        let timer = DispatchSource.makeTimerSource(queue: recordingQueue)
        timer.schedule(deadline: .now() + interval,
                       repeating: interval,
                       leeway: .milliseconds(Int(interval * 100)))
        timer.setEventHandler { [weak self] in
            self?.sampleAndWrite()
        }
        timer.resume()
        recordingTimer = timer

        // Kick off the calibration wizard. It runs purely as a UI state
        // machine — the recording timer above keeps writing rows throughout.
        // When the last position finishes it sets calibrationPhase = .idle
        // and the night recording continues uninterrupted in the same file.
        startCalibrationWizard()
    }

    func stopRecording() {
        guard isRecording else { return }

        // If the user stops during calibration (unusual but possible),
        // clean up the wizard state too.
        calibrationTimer?.invalidate()
        calibrationTimer = nil
        calibrationPhase = .idle

        // Stop the timer first so no more rows get queued
        recordingTimer?.cancel()
        recordingTimer = nil

        // Stop the silent audio loop. Releases the audio session so iOS
        // doesn't keep the audio subsystem warm overnight when not needed.
        audioKeeper.stop()

        // Stop watchdog before tearing down the recording state.
        stopWatchdog()

        // Release the background task so iOS can deactivate cleanly.
        endBackgroundTask()

        // Cancel any in-flight reconnect attempts — they may try to make
        // BLE calls during stop and contribute to UI hangs.
        reconnectingMacs.removeAll()

        // Capture state for the post-close UI update, then flip the public
        // flags immediately so the UI un-stuck right away. The actual file
        // close happens on a background queue — even an 880k-row CSV flush
        // shouldn't block the main thread.
        let urlToShare = currentSessionURL
        let handle = csvHandle
        csvHandle = nil
        currentSessionURL = nil
        sessionStartedAt = nil
        isRecording = false
        lastExportURL = urlToShare

        DispatchQueue.global(qos: .userInitiated).async {
            do {
                try handle?.synchronize()  // ensure all data is flushed to disk
                try handle?.close()
                print("Recording closed: \(urlToShare?.lastPathComponent ?? "?")")
            } catch {
                print("Error closing CSV: \(error)")
            }
        }
    }

    /// On app launch, look for the most recent CSV in Documents and expose
    /// it as `lastExportURL` so the user can share it even after a crash
    /// or force-quit during the previous session's stopRecording().
    func recoverLastSession() {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        guard let files = try? FileManager.default.contentsOfDirectory(
            at: docs, includingPropertiesForKeys: [.contentModificationDateKey]
        ) else { return }
        let csvs = files.filter { $0.lastPathComponent.hasPrefix("twitch_") &&
                                  $0.pathExtension == "csv" }
        let mostRecent = csvs.max { a, b in
            let ad = (try? a.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
            let bd = (try? b.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
            return ad < bd
        }
        if let recovered = mostRecent, lastExportURL == nil {
            DispatchQueue.main.async {
                self.lastExportURL = recovered
                print("Recovered previous session: \(recovered.lastPathComponent)")
            }
        }
    }

    /// Request a background task. iOS grants ~30 seconds of guaranteed
    /// background time. The expirationHandler runs ~5 seconds before the
    /// task expires; we use it to renew so the task is effectively
    /// open-ended for the duration of recording.
    private func beginBackgroundTask() {
        endBackgroundTask()  // end any existing task first
        bgTaskID = UIApplication.shared.beginBackgroundTask(withName: "twitch-recording") { [weak self] in
            // Renew before iOS forces expiration. This keeps us in the
            // "needs background time" pool indefinitely.
            self?.beginBackgroundTask()
        }
    }

    private func endBackgroundTask() {
        if bgTaskID != .invalid {
            UIApplication.shared.endBackgroundTask(bgTaskID)
            bgTaskID = .invalid
        }
    }

    /// Independent watchdog timer fired on the main run loop every 25
    /// seconds. Two jobs:
    ///   1. Renew the background task. The expirationHandler approach in
    ///      beginBackgroundTask() should work, but iOS sometimes doesn't
    ///      call it in time. Renewing eagerly is safer.
    ///   2. Detect a stalled recording timer. If the sample count hasn't
    ///      increased for two consecutive watchdog ticks (~50 seconds),
    ///      the DispatchSourceTimer has stopped firing — restart it.
    private func startWatchdog() {
        stopWatchdog()
        lastWatchdogSampleCount = backgroundSampleCount
        watchdogStallTicks = 0
        watchdogTimer = Timer.scheduledTimer(withTimeInterval: 25.0, repeats: true) { [weak self] _ in
            self?.watchdogTick()
        }
        // Allow the timer to fire even when the UI is in tracking-event mode
        // (rare but worth being defensive)
        RunLoop.main.add(watchdogTimer!, forMode: .common)
    }

    private func stopWatchdog() {
        watchdogTimer?.invalidate()
        watchdogTimer = nil
    }

    private func watchdogTick() {
        // 1. Renew the background task
        beginBackgroundTask()

        // 2. Ensure audio session is still active. Another app (Eight Sleep,
        //    Calm, etc.) may have stolen it without sending an interruption
        //    notification while we were backgrounded. reclaimAudioSession()
        //    is a no-op if already playing.
        if isRecording {
            audioKeeper.reclaimAudioSession()
        }

        // 3. Check the recording timer is alive
        guard isRecording else { return }
        let progress = backgroundSampleCount - lastWatchdogSampleCount
        if progress == 0 {
            watchdogStallTicks += 1
            print("WATCHDOG: recording timer hasn't ticked in \(watchdogStallTicks * 25)s")
            if watchdogStallTicks >= 2 {
                // 50+ seconds with no samples — recording timer is stalled.
                // Restart it.
                print("WATCHDOG: restarting stalled recording timer")
                restartRecordingTimer()
                watchdogStallTicks = 0
            }
        } else {
            watchdogStallTicks = 0
        }
        lastWatchdogSampleCount = backgroundSampleCount
    }

    /// Recreate the DispatchSourceTimer without touching the file handle
    /// or recording state. Used by the watchdog when the existing timer
    /// has stopped firing under iOS throttling.
    /// IMPORTANT: reuses recordingQueue (single instance) — never creates
    /// a new DispatchQueue here, which would leak a kernel queue + thread
    /// pool per restart and cause OOM after many hours of recording.
    private func restartRecordingTimer() {
        recordingTimer?.cancel()
        let interval = 1.0 / Self.recordingHz
        let timer = DispatchSource.makeTimerSource(queue: recordingQueue)
        timer.schedule(deadline: .now() + interval,
                       repeating: interval,
                       leeway: .milliseconds(Int(interval * 100)))
        timer.setEventHandler { [weak self] in
            self?.sampleAndWrite()
        }
        timer.resume()
        recordingTimer = timer
    }

    private func filenameStamp(_ date: Date) -> String {
        let f = DateFormatter()
        f.dateFormat = "yyyyMMdd_HHmmss"
        return f.string(from: date)
    }

    // MARK: -- Live UI refresh (5 Hz) -------------------------------------

    private func startRefreshThread() {
        Thread(target: self, selector: #selector(refreshView), object: nil).start()
    }

    @objc private func refreshView() {
        // IMPORTANT: this thread MUST NOT use DispatchQueue.main.sync.
        // Doing so risks a deadlock if the main thread is ever waiting on
        // anything that, transitively, needs this thread. iOS terminates
        // unresponsive apps with the watchdog timer (~5s), and we suspect
        // this was a cause of mid-night kills.
        //
        // The device references are stored as @Published — they can
        // technically be mutated from the main thread while we read here,
        // but for a single property load (atomic on iOS for reference
        // types), the worst-case race is a stale value displayed for one
        // 200ms cycle. That's invisible to humans.
        while true {
            Thread.sleep(forTimeInterval: 0.2)
            let rDevice = self.rightAnkle.device
            let lDevice = self.leftAnkle.device

            let rSnap = makeSnapshot(rDevice)
            let lSnap = makeSnapshot(lDevice)

            DispatchQueue.main.async {
                self.rightAnkle.liveSnapshot = rSnap.text
                self.rightAnkle.lastBattery = rSnap.battery
                self.leftAnkle.liveSnapshot = lSnap.text
                self.leftAnkle.lastBattery = lSnap.battery
            }
        }
    }

    private func makeSnapshot(_ device: Bwt901ble?) -> (text: String, battery: String) {
        guard let d = device else { return ("Not connected", "—") }
        let ax = d.getDeviceData(WitSensorKey.AccX) ?? "—"
        let ay = d.getDeviceData(WitSensorKey.AccY) ?? "—"
        let az = d.getDeviceData(WitSensorKey.AccZ) ?? "—"
        let bat = d.getDeviceData(WitSensorKey.ElectricQuantityPercentage) ?? "—"
        let text = "AX: \(ax)\nAY: \(ay)\nAZ: \(az)"
        return (text, bat)
    }
}


// **********************************************************
// MARK: MainView — single-screen UI for two sensors
// **********************************************************
struct MainView: View {

    @EnvironmentObject var ctx: AppContext
    @State private var showShareSheet = false

    var body: some View {
        ScrollView {
            VStack(spacing: 16) {
                // Two slot cards side-by-side on iPad-ish widths,
                // stacked on phone widths.
                SensorCard(slot: ctx.rightAnkle)
                SensorCard(slot: ctx.leftAnkle)
                // HRMCard hidden — chest strap retired due to skin irritation.
                // Code retained in case we re-add HR data later.

                // Calibration wizard — shown during the opening phase of a recording.
                // Replaces the normal recording panel until all 3 positions are done.
                if ctx.calibrationPhase.isActive {
                    CalibrationCard()
                }

                // Discovered devices (only shows if user is scanning)
                if ctx.enableScan || !ctx.deviceList.isEmpty {
                    GroupBox(label: Label("Discovered devices", systemImage: "antenna.radiowaves.left.and.right")) {
                        VStack(spacing: 8) {
                            Button(ctx.enableScan ? "Stop scanning" : "Scan for sensors") {
                                ctx.enableScan ? ctx.stopScan() : ctx.scanDevices()
                            }
                            .buttonStyle(.bordered)

                            ForEach(ctx.deviceList) { device in
                                Button {
                                    ctx.requestAssignment(device)
                                } label: {
                                    HStack {
                                        Text(device.name ?? "(unnamed)")
                                        Spacer()
                                        Text(device.mac?.prefix(8) ?? "")
                                            .font(.caption)
                                            .foregroundColor(.secondary)
                                    }
                                }
                                .buttonStyle(.bordered)
                            }
                        }
                        .frame(maxWidth: .infinity)
                    }
                } else if ctx.connectedSlots.count < 2 {
                    Button("Scan for sensors") {
                        ctx.scanDevices()
                    }
                    .buttonStyle(.bordered)
                }

                // Recording panel
                GroupBox(label: Label("Recording", systemImage: "record.circle")) {
                    VStack(spacing: 12) {
                        if ctx.isRecording {
                            Text("RECORDING")
                                .font(.title2.bold())
                                .foregroundColor(.red)
                            Text("\(ctx.sampleCount) samples @ \(Int(AppContext.recordingHz)) Hz")
                                .font(.system(.body, design: .monospaced))
                            Text("\(ctx.connectedSlots.count) sensor(s) active")
                                .font(.caption)
                                .foregroundColor(.secondary)
                            BackgroundAudioStatusView(keeper: ctx.audioKeeper)
                            if let started = ctx.sessionStartedAt {
                                Text("Started \(started, style: .time)")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                            }
                            Button("Stop recording") {
                                ctx.stopRecording()
                            }
                            .buttonStyle(.borderedProminent)
                            .tint(.red)
                        } else if ctx.calibrationPhase.isActive {
                            // Hidden during calibration — CalibrationCard is shown instead
                            EmptyView()
                        } else {
                            Button("Start recording") {
                                ctx.startRecording()
                            }
                            .buttonStyle(.borderedProminent)
                            .disabled(!ctx.hasAnyDataSource)
                            if !ctx.hasAnyDataSource {
                                Text("Connect at least one sensor or HRM first")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                            }
                        }
                    }
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 4)
                }

                // Export panel
                if let url = ctx.lastExportURL, !ctx.isRecording {
                    GroupBox(label: Label("Last session", systemImage: "doc.text")) {
                        VStack(spacing: 8) {
                            Text(url.lastPathComponent)
                                .font(.caption)
                                .foregroundColor(.secondary)
                            Button("Share CSV") {
                                showShareSheet = true
                            }
                            .buttonStyle(.bordered)
                        }
                        .frame(maxWidth: .infinity)
                    }
                    .sheet(isPresented: $showShareSheet) {
                        ShareSheet(items: [url])
                    }
                }
            }
            .padding()
        }
        .navigationTitle("Twitch Recorder")
        // Role picker for pending device
        .confirmationDialog(
            "Assign role",
            isPresented: Binding(
                get: { ctx.pendingDevice != nil },
                set: { if !$0 { ctx.cancelAssignment() } }
            ),
            titleVisibility: .visible,
            presenting: ctx.pendingDevice
        ) { _ in
            Button("Right ankle") { ctx.assignPending(to: .rightAnkle) }
            Button("Left ankle") { ctx.assignPending(to: .leftAnkle) }
            Button("Cancel", role: .cancel) { ctx.cancelAssignment() }
        } message: { device in
            Text("Assign \(device.name ?? "this device") to:")
        }
        .onAppear {
            UIApplication.shared.isIdleTimerDisabled = true
        }
        .onDisappear {
            UIApplication.shared.isIdleTimerDisabled = false
        }
    }
}


// **********************************************************
// MARK: CalibrationCard — position calibration wizard
// **********************************************************
//
// Shown only while a recording is active and calibrationPhase != .idle.
// The wizard runs as the opening ~33 seconds of the night file — same CSV,
// same recording timer. When done it sets calibrationPhase = .idle and
// the card disappears, replaced by the normal "RECORDING" panel.
struct CalibrationCard: View {
    @EnvironmentObject var ctx: AppContext

    var body: some View {
        GroupBox(label: Label("Position calibration", systemImage: "figure.stand")) {
            VStack(spacing: 12) {
                switch ctx.calibrationPhase {

                case .idle:
                    // Should not be visible — MainView only shows this card when isActive
                    EmptyView()

                case .countdown(let position, let secondsLeft):
                    Text("Get ready…")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Text(positionLabel(position))
                        .font(.system(size: 36, weight: .bold, design: .rounded))
                        .foregroundColor(.orange)
                    Text("Starting in \(secondsLeft)s")
                        .font(.title3.monospacedDigit())
                        .foregroundColor(.orange)
                    positionIcon(position)
                        .font(.system(size: 48))
                    cancelButton

                case .holding(let position, let secondsLeft):
                    Text("Hold still")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Text(positionLabel(position))
                        .font(.system(size: 36, weight: .bold, design: .rounded))
                        .foregroundColor(.green)
                    ZStack {
                        Circle()
                            .stroke(Color.green.opacity(0.2), lineWidth: 8)
                        Circle()
                            .trim(from: 0,
                                  to: CGFloat(AppContext.calibrationHoldSeconds - secondsLeft + 1)
                                    / CGFloat(AppContext.calibrationHoldSeconds))
                            .stroke(Color.green, style: StrokeStyle(lineWidth: 8, lineCap: .round))
                            .rotationEffect(.degrees(-90))
                            .animation(.linear(duration: 1), value: secondsLeft)
                        Text("\(secondsLeft)s")
                            .font(.system(size: 28, weight: .semibold, design: .monospaced))
                    }
                    .frame(width: 90, height: 90)
                    positionIcon(position)
                        .font(.system(size: 48))
                    cancelButton
                }
            }
            .frame(maxWidth: .infinity)
            .padding(.vertical, 4)
        }
    }

    private var cancelButton: some View {
        Button("Cancel recording", role: .cancel) {
            ctx.cancelCalibration()
        }
        .buttonStyle(.bordered)
        .foregroundColor(.secondary)
        .font(.caption)
    }

    private func positionLabel(_ pos: String) -> String {
        switch pos {
        case "back":  return "On your back"
        case "right": return "Right side"
        case "left":  return "Left side"
        default:      return pos.capitalized
        }
    }

    private func positionIcon(_ pos: String) -> Text {
        switch pos {
        case "back":  return Text("🛌")
        case "right": return Text("➡️")
        case "left":  return Text("⬅️")
        default:      return Text("🛌")
        }
    }
}


// **********************************************************
// MARK: SensorCard — one sensor slot with controls
// **********************************************************
struct SensorCard: View {
    @ObservedObject var slot: SensorSlot
    @EnvironmentObject var ctx: AppContext

    var body: some View {
        GroupBox(label: Label(slot.role.label, systemImage: roleIcon)) {
            if slot.isConnected {
                VStack(alignment: .leading, spacing: 6) {
                    HStack {
                        Text(slot.device?.name ?? "WT901BLECL")
                            .font(.headline)
                        Spacer()
                        Text("🔋 \(slot.lastBattery)%")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                    Text(slot.liveSnapshot)
                        .font(.system(.caption, design: .monospaced))
                        .foregroundColor(.secondary)
                    HStack(spacing: 8) {
                        // Sensor output 50 Hz feeds our 30 Hz CSV polling —
                        // BLE bandwidth caps real throughput around 24 Hz, so
                        // sensor at 50 keeps the poll pipe full. Output rate
                        // persists in sensor NV memory; only need to set once
                        // per sensor lifetime.
                        Button("Set sensor → 50 Hz") {
                            ctx.setSensorOutputRate(0x08, for: slot.device)
                        }
                        .buttonStyle(.bordered)
                        Button("Disconnect") {
                            ctx.disconnect(slot)
                        }
                        .buttonStyle(.bordered)
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
            } else {
                Text("No \(slot.role.label.lowercased()) sensor connected")
                    .foregroundColor(.secondary)
                    .frame(maxWidth: .infinity)
            }
        }
    }

    private var roleIcon: String {
        // Both roles are ankles; distinguished by side label.
        slot.role == .rightAnkle ? "figure.walk" : "figure.walk.arrival"
    }
}


// **********************************************************
// MARK: HRMCard — heart-rate strap UI
// **********************************************************
//
// IMPORTANT: HRMManager is observed via @ObservedObject (not via the
// parent AppContext) because SwiftUI only re-renders a view when the
// objects it directly observes publish a change. AppContext's
// @Published var hrm only fires on reference changes (which never
// happen), so we'd miss every per-heartbeat update otherwise.
struct HRMCard: View {
    @ObservedObject var hrm: HRMManager

    var body: some View {
        GroupBox(label: Label("Heart rate", systemImage: "heart.fill")) {
            if hrm.connected != nil {
                VStack(alignment: .leading, spacing: 8) {
                    HStack {
                        Text(hrm.connected?.name ?? "HRM")
                            .font(.headline)
                        Spacer()
                        if let bat = hrm.lastBattery {
                            Text("🔋 \(bat)%")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                    }
                    // Big current BPM (grayed out when stale)
                    HStack(alignment: .firstTextBaseline, spacing: 6) {
                        if let hr = hrm.lastHR, !hrm.hrIsStale {
                            Text("\(hr)")
                                .font(.system(size: 44, weight: .semibold, design: .rounded))
                                .foregroundColor(.red)
                        } else if hrm.lastHR != nil && hrm.hrIsStale {
                            // Stale value — show last known but grayed and crossed out
                            Text("\(hrm.lastHR!)")
                                .font(.system(size: 44, weight: .semibold, design: .rounded))
                                .foregroundColor(.secondary)
                                .strikethrough()
                        } else {
                            Text("—")
                                .font(.system(size: 44, weight: .semibold, design: .rounded))
                                .foregroundColor(.secondary)
                        }
                        Text("bpm")
                            .font(.title3)
                            .foregroundColor(.secondary)
                        Spacer()
                        if let rr = hrm.lastRR, !hrm.hrIsStale {
                            VStack(alignment: .trailing, spacing: 2) {
                                Text("RR")
                                    .font(.caption2)
                                    .foregroundColor(.secondary)
                                Text("\(rr) ms")
                                    .font(.system(.caption, design: .monospaced))
                                    .foregroundColor(.secondary)
                            }
                        }
                    }
                    if hrm.hrIsStale {
                        Text("⚠️ no signal — strap off skin or BLE timeout")
                            .font(.caption)
                            .foregroundColor(.orange)
                    } else if hrm.rrIsStale {
                        // HR is being reported but RR isn't — strap can't
                        // confidently detect individual beats. Usually means
                        // dry/loose electrodes.
                        Text("⚠️ low confidence — wet the electrodes for better contact")
                            .font(.caption)
                            .foregroundColor(.orange)
                    }
                    Button("Disconnect") {
                        hrm.disconnect()
                    }
                    .buttonStyle(.bordered)
                }
                .frame(maxWidth: .infinity, alignment: .leading)
            } else {
                VStack(spacing: 8) {
                    Button(hrm.enableScan ? "Stop scanning" : "Scan for HRM") {
                        hrm.enableScan ? hrm.stopScan() : hrm.scan()
                    }
                    .buttonStyle(.bordered)
                    ForEach(hrm.discovered, id: \.identifier) { peripheral in
                        Button {
                            hrm.connect(peripheral)
                        } label: {
                            HStack {
                                Text(peripheral.name ?? "(unnamed)")
                                Spacer()
                                Text(peripheral.identifier.uuidString.prefix(8))
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                            }
                        }
                        .buttonStyle(.bordered)
                    }
                }
                .frame(maxWidth: .infinity)
            }
        }
    }
}


// **********************************************************
// MARK: ShareSheet — UIKit bridge for UIActivityViewController
// **********************************************************
// **********************************************************
// MARK: BackgroundAudioStatusView — visible indicator
// **********************************************************
//
// Live status of the silent-audio background-keep-alive trick. Shown in
// the recording panel so you can see at a glance whether the app's
// background-stay-alive mechanism is working.
//
// IMPORTANT: this view directly observes BackgroundAudioKeeper (not via
// AppContext) for the same SwiftUI re-render reason that HRMCard does —
// AppContext only fires on @Published reference changes, not on changes
// to nested object properties.
struct BackgroundAudioStatusView: View {
    @ObservedObject var keeper: BackgroundAudioKeeper

    var body: some View {
        if keeper.isActive {
            HStack(spacing: 4) {
                Image(systemName: "speaker.wave.2.fill")
                    .font(.caption2)
                Text("background audio active (1 Hz tone, inaudible)")
                    .font(.caption2)
            }
            .foregroundColor(.green)
        } else if let err = keeper.lastError {
            HStack(spacing: 4) {
                Image(systemName: "exclamationmark.triangle.fill")
                    .font(.caption2)
                Text("audio: \(err)")
                    .font(.caption2)
                    .lineLimit(1)
            }
            .foregroundColor(.orange)
        } else {
            HStack(spacing: 4) {
                Image(systemName: "speaker.slash")
                    .font(.caption2)
                Text("background audio off")
                    .font(.caption2)
            }
            .foregroundColor(.secondary)
        }
    }
}


struct ShareSheet: UIViewControllerRepresentable {
    let items: [Any]
    func makeUIViewController(context: Context) -> UIActivityViewController {
        UIActivityViewController(activityItems: items, applicationActivities: nil)
    }
    func updateUIViewController(_ controller: UIActivityViewController, context: Context) {}
}
